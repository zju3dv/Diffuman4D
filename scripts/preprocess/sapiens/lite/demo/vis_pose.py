# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os, os.path as osp
import time
import cv2
import json_tricks as json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import ArgumentParser
from collections import defaultdict
from typing import List, Optional, Sequence, Union
from tqdm import tqdm
from glob import glob

from adhoc_image_dataset import AdhocImageDataset
from classes_and_palettes import (
    COCO_KPTS_COLORS,
    COCO_WHOLEBODY_KPTS_COLORS,
    GOLIATH_KPTS_COLORS,
    GOLIATH_SKELETON_INFO,
    COCO_SKELETON_INFO,
    COCO_WHOLEBODY_SKELETON_INFO,
)
from pose_utils import nms, top_down_affine_transform, udp_decode

try:
    from mmdet.apis import inference_detector, init_detector
    from mmdet.structures import DetDataSample, SampleList
    from mmdet.utils import get_test_pipeline_cfg

    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


def preprocess_pose(orig_img, bboxes_list, input_shape, mean, std):
    """Preprocess pose images and bboxes."""
    preprocessed_images = []
    centers = []
    scales = []
    for bbox in bboxes_list:
        img, center, scale = top_down_affine_transform(orig_img.copy(), bbox)
        img = cv2.resize(img, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
        img = torch.from_numpy(img)
        img = img[[2, 1, 0], ...].float()
        mean = torch.Tensor(mean).view(-1, 1, 1)
        std = torch.Tensor(std).view(-1, 1, 1)
        img = (img - mean) / std
        preprocessed_images.append(img)
        centers.extend(center)
        scales.extend(scale)
    return preprocessed_images, centers, scales


def batch_inference_topdown(
    model: nn.Module, imgs: List[Union[np.ndarray, str]], dtype=torch.bfloat16, device="cuda", flip=False
):
    imgs = imgs.to(dtype).to(device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
        heatmaps = model(imgs)
        if flip:
            heatmaps_ = model(imgs.flip(-1))
            heatmaps = (heatmaps + heatmaps_) * 0.5
    return heatmaps.cpu()


def img_save_and_vis(
    img,
    results,
    output_path,
    input_shape,
    heatmap_scale,
    kpt_colors,
    kpt_thr,
    radius,
    skeleton_info,
    thickness,
    save_image,
):
    # pred_instances_list = split_instances(result)
    heatmap = results["heatmaps"]
    centres = results["centres"]
    scales = results["scales"]
    img_shape = img.shape
    instance_keypoints = []
    instance_scores = []
    # print(scales[0], centres[0])
    for i in range(len(heatmap)):
        result = udp_decode(
            heatmap[i].cpu().unsqueeze(0).float().data[0].numpy(),
            input_shape,
            (int(input_shape[0] / heatmap_scale), int(input_shape[1] / heatmap_scale)),
        )

        keypoints, keypoint_scores = result
        keypoints = (keypoints / input_shape) * scales[i] + centres[i] - 0.5 * scales[i]
        instance_keypoints.append(keypoints[0])
        instance_scores.append(keypoint_scores[0])

    pred_save_path = output_path.replace(".jpg", ".json").replace(".webp", ".json").replace(".png", ".json")
    os.makedirs(osp.dirname(pred_save_path), exist_ok=True)
    with open(pred_save_path, "w") as f:
        json.dump(
            dict(
                instance_info=[
                    {
                        "keypoints": keypoints.tolist(),
                        "keypoint_scores": keypoint_scores.tolist(),
                    }
                    for keypoints, keypoint_scores in zip(instance_keypoints, instance_scores)
                ]
            ),
            f,
            indent="\t",
        )

    if not save_image:
        return

    # ? draw on grey background. it will be mapped to 0.0 in diffusion models
    img = np.ones_like(img) * 0.5
    # img = pyvips.Image.new_from_array(img)
    instance_keypoints = np.array(instance_keypoints).astype(np.float32)
    instance_scores = np.array(instance_scores).astype(np.float32)

    keypoints_visible = np.ones(instance_keypoints.shape[:-1])
    for kpts, score, visible in zip(instance_keypoints, instance_scores, keypoints_visible):
        kpts = np.array(kpts, copy=False)

        if kpt_colors is None or isinstance(kpt_colors, str) or len(kpt_colors) != len(kpts):
            raise ValueError(
                f"the length of kpt_color " f"({len(kpt_colors)}) does not matches " f"that of keypoints ({len(kpts)})"
            )

        # draw skeleton
        for skid, link_info in skeleton_info.items():
            pt1_idx, pt2_idx = link_info["link"]
            color = link_info["color"][::-1]  # BGR

            pt1 = kpts[pt1_idx]
            pt1_score = score[pt1_idx]
            pt2 = kpts[pt2_idx]
            pt2_score = score[pt2_idx]

            if pt1_score > kpt_thr and pt2_score > kpt_thr:
                x1_coord = int(pt1[0])
                y1_coord = int(pt1[1])
                x2_coord = int(pt2[0])
                y2_coord = int(pt2[1])
                cv2.line(
                    img,
                    (x1_coord, y1_coord),
                    (x2_coord, y2_coord),
                    color,
                    thickness=thickness,
                )

        # draw each point on image
        for kid, kpt in enumerate(kpts):
            if score[kid] < kpt_thr or not visible[kid] or kpt_colors[kid] is None:
                # skip the point that should not be drawn
                continue

            color = kpt_colors[kid]
            if not isinstance(color, str):
                color = tuple(int(c) for c in color[::-1])
            img = cv2.circle(img, (int(kpt[0]), int(kpt[1])), int(radius), color, -1)

    os.makedirs(osp.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)


def fake_pad_images_to_batchsize(imgs, batch_size):
    return F.pad(imgs, (0, 0, 0, 0, 0, 0, 0, batch_size - imgs.shape[0]), value=0)


def load_model(checkpoint, use_torchscript=False):
    if use_torchscript:
        return torch.jit.load(checkpoint)
    else:
        return torch.export.load(checkpoint).module()


def main():
    """Visualize the demo images.
    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument("pose_checkpoint", help="Checkpoint file for pose")
    parser.add_argument("--det-config", default="", help="Config file for detection")
    parser.add_argument("--det-checkpoint", default="", help="Checkpoint file for detection")
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--fmasks_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--num_keypoints",
        type=int,
        default=133,
        help="Number of keypoints in the pose model. Used for visualization",
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs="+",
        default=[1024, 768],
        help="input image size (height, width)",
    )
    parser.add_argument(
        "--batch_size",
        "--batch-size",
        type=int,
        default=1,
        help="Set batch size to do batch inference.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Set number of workers per GPU",
    )
    parser.add_argument("--gpu_ids", default="0,1", help="Device used for inference")
    parser.add_argument("--fp16", action="store_true", default=False, help="Model inference dtype")
    parser.add_argument(
        "--det-cat-id",
        type=int,
        default=0,
        help="Category id for bounding box detection model",
    )
    parser.add_argument("--bbox-thr", type=float, default=0.3, help="Bounding box score threshold")
    parser.add_argument("--nms-thr", type=float, default=0.3, help="IoU threshold for bounding box NMS")
    parser.add_argument("--kpt-thr", type=float, default=0.3, help="Visualizing keypoint thresholds")
    parser.add_argument("--radius", type=int, default=9, help="Keypoint radius for visualization")
    parser.add_argument(
        "--thickness",
        type=int,
        default=-1,
        help="Keypoint skeleton thickness for visualization",
    )
    parser.add_argument(
        "--heatmap-scale",
        type=int,
        default=4,
        help="Heatmap scale for keypoints. Image to heatmap ratio",
    )
    parser.add_argument(
        "--flip",
        type=bool,
        default=False,
        help="Flip the input image horizontally and inference again",
    )
    parser.add_argument("--image_ext", type=str, default=".jpg", help="Image/keypoints extension")
    parser.add_argument(
        "--save_image",
        action="store_true",
        default=False,
        help="Whether to save keypoint maps",
    )
    parser.add_argument(
        "--skip_exists",
        action="store_true",
        help="Whether to skip the existing keypoints",
    )

    args = parser.parse_args()

    if args.det_config is None or args.det_config == "":
        use_det = False
    else:
        use_det = True
        assert has_mmdet, "Please install mmdet to run the demo."
        assert args.det_checkpoint is not None

        from detector_utils import (
            adapt_mmdet_pipeline,
            init_detector,
            process_images_detector,
        )

    assert args.images_dir != ""
    ## if skeleton thickness is not specified, use radius as thickness
    if args.thickness == -1:
        args.thickness = args.radius

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3,) + tuple(args.shape)
    else:
        raise ValueError("invalid input shape")

    torch._inductor.config.force_fuse_int_mm_with_mul = True
    torch._inductor.config.use_mixed_mm = True
    args.gpu_ids = [int(i) for i in args.gpu_ids.split(",")]
    num_gpus = len(args.gpu_ids)
    args.num_workers = args.num_workers * num_gpus

    detectors = []
    pose_estimators = []

    for i in args.gpu_ids:
        # build detector
        if use_det:
            detector = init_detector(args.det_config, args.det_checkpoint, device=f"cuda:{i}")
            detector.cfg = adapt_mmdet_pipeline(detector.cfg)
            detectors.append(detector)

        # build pose estimator
        USE_TORCHSCRIPT = "_torchscript" in args.pose_checkpoint
        # build the model from a checkpoint file
        pose_estimator = load_model(args.pose_checkpoint, USE_TORCHSCRIPT)
        ## no precision conversion needed for torchscript. run at fp32
        if not USE_TORCHSCRIPT:
            dtype = torch.half if args.fp16 else torch.bfloat16
            pose_estimator.to(dtype)
            pose_estimator = torch.compile(pose_estimator, mode="max-autotune", fullgraph=True)
        else:
            dtype = torch.float32  # TorchScript models use float32
            pose_estimator = pose_estimator.to(device=f"cuda:{i}")
        pose_estimators.append(pose_estimator)

    # hard code the image extension
    image_paths = sorted(
        glob(f"{args.images_dir}/**/*.jpg", recursive=True) + glob(f"{args.images_dir}/**/*.webp", recursive=True)
    )
    if args.fmasks_dir is not None and args.fmasks_dir != "None":
        fmask_paths = sorted(glob(f"{args.fmasks_dir}/**/*.png", recursive=True))
    else:
        fmask_paths = None

    assert fmask_paths is None or len(image_paths) == len(
        fmask_paths
    ), "image_paths and fmask_paths must have the same length to enable background removal"

    scale = args.heatmap_scale
    # do not provide preprocess args for detector as we use mmdet
    inference_dataset = AdhocImageDataset(image_paths, fmask_paths)

    KPTS_COLORS = COCO_WHOLEBODY_KPTS_COLORS  ## 133 keypoints
    SKELETON_INFO = COCO_WHOLEBODY_SKELETON_INFO

    if args.num_keypoints == 17:
        KPTS_COLORS = COCO_KPTS_COLORS
        SKELETON_INFO = COCO_SKELETON_INFO
    elif args.num_keypoints == 308:
        KPTS_COLORS = GOLIATH_KPTS_COLORS
        SKELETON_INFO = GOLIATH_SKELETON_INFO

    def process_single_batch(idx):
        # get model and data
        gpu_idx = idx % num_gpus
        device = f"cuda:{args.gpu_ids[gpu_idx]}"
        detector = detectors[gpu_idx]
        pose_estimator = pose_estimators[gpu_idx]

        image_path, orig_img = inference_dataset[idx]

        # ? we add the cam_label here to fit the easyvolcap data format
        output_path = osp.join(args.output_dir, "/".join(image_path.split("/")[-2:]))
        args.image_ext = osp.splitext(image_path)[1]
        output_json_path = output_path.replace(args.image_ext, ".json")
        if args.skip_exists and osp.exists(output_json_path):
            try:
                json.load(open(output_json_path))
                return
            except Exception as e:
                print(f"Error reading {output_json_path}: {e}")

        batch_output_paths = [output_path]
        batch_orig_imgs = orig_img[None, ...]

        orig_img_shape = batch_orig_imgs.shape
        valid_images_len = len(batch_orig_imgs)
        if use_det:
            imgs = batch_orig_imgs.copy()[..., [2, 1, 0]]
            bboxes_batch = process_images_detector(args, imgs, detector)
        else:
            bboxes_batch = [[] for _ in range(len(batch_orig_imgs))]

        assert len(bboxes_batch) == valid_images_len

        for i, bboxes in enumerate(bboxes_batch):
            if len(bboxes) == 0:
                bboxes_batch[i] = np.array([[0, 0, orig_img_shape[1], orig_img_shape[2]]])  # orig_img_shape: B H W C

        img_bbox_map = {}
        for i, bboxes in enumerate(bboxes_batch):
            img_bbox_map[i] = len(bboxes)

        args_list = [
            (
                orig_img,
                bbox_list,
                (input_shape[1], input_shape[2]),
                [123.5, 116.5, 103.5],
                [58.5, 57.0, 57.5],
            )
            for orig_img, bbox_list in zip(batch_orig_imgs, bboxes_batch)
        ]

        pose_ops = []
        for _args in args_list:
            pose_op = preprocess_pose(*_args)
            pose_ops.append(pose_op)

        pose_imgs, pose_img_centers, pose_img_scales = [], [], []
        for op in pose_ops:
            pose_imgs.extend(op[0])
            pose_img_centers.extend(op[1])
            pose_img_scales.extend(op[2])

        args.batch_size = 1  # process one image at a time
        n_pose_batches = (len(pose_imgs) + args.batch_size - 1) // args.batch_size

        # use this to tell torch compiler the start of model invocation as in 'flip' mode the tensor output is overwritten
        torch.compiler.cudagraph_mark_step_begin()
        pose_results = []
        for i in range(n_pose_batches):
            imgs = torch.stack(pose_imgs[i * args.batch_size : (i + 1) * args.batch_size], dim=0)
            valid_len = len(imgs)
            imgs = fake_pad_images_to_batchsize(imgs, args.batch_size)
            pose_results.extend(batch_inference_topdown(pose_estimator, imgs, dtype=dtype, device=device)[:valid_len])

        batched_results = []
        for _, bbox_len in img_bbox_map.items():
            result = {
                "heatmaps": pose_results[:bbox_len].copy(),
                "centres": pose_img_centers[:bbox_len].copy(),
                "scales": pose_img_scales[:bbox_len].copy(),
            }
            batched_results.append(result)
            del (
                pose_results[:bbox_len],
                pose_img_centers[:bbox_len],
                pose_img_scales[:bbox_len],
            )

        assert len(batched_results) == len(batch_orig_imgs)

        args_list = [
            (
                img,
                res,
                output_path,
                (input_shape[2], input_shape[1]),
                scale,
                KPTS_COLORS,
                args.kpt_thr,
                args.radius,
                SKELETON_INFO,
                args.thickness,
                args.save_image,
            )
            for img, res, output_path in zip(
                batch_orig_imgs[:valid_images_len],
                batched_results[:valid_images_len],
                batch_output_paths,
            )
        ]
        for _args in args_list:
            img_save_and_vis(*_args)

    from easyvolcap.utils.parallel_utils import parallel_execution

    print(f"Predicting 2D keypoints using Sapiens Lite on {len(inference_dataset)} images")
    print(f"The results will be saved to: {args.output_dir}")
    parallel_execution(
        list(range(len(inference_dataset))),
        action=process_single_batch,
        num_workers=args.num_workers,
        print_progress=True,
        sequential=False,
        desc="Predicting 2D keypoints using Sapiens Lite",
    )


if __name__ == "__main__":
    main()
