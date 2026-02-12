# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os, os.path as osp
import time
import threading
import cv2
import json_tricks as json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import fire
from collections import defaultdict
from typing import List, Optional, Sequence, Union
from tqdm import tqdm
from glob import glob

from adhoc_image_dataset import AdhocImageDataset
from adhoc_video_dataset import AdhocVideoDataset

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
    # output_size = (width, height) matching the model input
    output_size = (input_shape[1], input_shape[0])
    for bbox in bboxes_list:
        img, center, scale = top_down_affine_transform(orig_img.copy(), bbox, output_size=output_size)
        img = cv2.resize(img, output_size, interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
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


def load_pose_estimators(pose_checkpoint, gpu_ids=None, dtype=torch.bfloat16):
    if gpu_ids is None:
        gpu_ids = list(range(torch.cuda.device_count()))

    pose_estimators = []
    for gpu_id in gpu_ids:
        USE_TORCHSCRIPT = "_torchscript" in pose_checkpoint
        # build the model from a checkpoint file
        pose_estimator = load_model(pose_checkpoint, USE_TORCHSCRIPT)
        ## no precision conversion needed for torchscript. run at fp32
        if not USE_TORCHSCRIPT:
            pose_estimator.to(dtype)
            pose_estimator = torch.compile(pose_estimator, mode="max-autotune", fullgraph=True)
        else:
            dtype = torch.float32  # TorchScript models use float32
            pose_estimator = pose_estimator.to(device=f"cuda:{gpu_id}")
        pose_estimators.append(pose_estimator)
    return pose_estimators


def load_detectors(det_config, det_checkpoint, gpu_ids=None):
    from detector_utils import init_detector, adapt_mmdet_pipeline

    if gpu_ids is None:
        gpu_ids = list(range(torch.cuda.device_count()))

    detectors = []
    for gpu_id in gpu_ids:
        # build detector
        detector = init_detector(det_config, det_checkpoint, device=f"cuda:{gpu_id}")
        detector.cfg = adapt_mmdet_pipeline(detector.cfg)
        detectors.append(detector)
    return detectors


def inference_sapiens_pose(
    pose_checkpoint="",
    det_config="",
    det_checkpoint="",
    pose_estimators=None,
    detectors=None,
    images_dir=None,
    video_path=None,
    fmasks_dir=None,
    output_dir=None,
    num_keypoints=133,
    shape=(1024, 768),
    batch_size=1,
    num_workers=16,
    gpu_ids=None,
    fp16=False,
    det_cat_id=0,
    bbox_thr=0.3,
    nms_thr=0.3,
    kpt_thr=0.3,
    radius=9,
    thickness=-1,
    heatmap_scale=4,
    flip=False,
    image_ext=".jpg",
    save_image=False,
    skip_exists=False,
):
    """Visualize the demo images.
    Using mmdet to detect the human.

    Args:
        pose_checkpoint: Checkpoint file for pose
        det_config: Config file for detection
        det_checkpoint: Checkpoint file for detection
        images_dir: Directory containing images
        video_path: Path to video file
        fmasks_dir: Directory containing foreground masks
        output_dir: Output directory (required)
        num_keypoints: Number of keypoints in the pose model. Used for visualization
        shape: Input image size (height, width)
        batch_size: Set batch size to do batch inference
        num_workers: Set number of workers per GPU
        gpu_ids: Device used for inference
        fp16: Model inference dtype
        det_cat_id: Category id for bounding box detection model
        bbox_thr: Bounding box score threshold
        nms_thr: IoU threshold for bounding box NMS
        kpt_thr: Visualizing keypoint thresholds
        radius: Keypoint radius for visualization
        thickness: Keypoint skeleton thickness for visualization
        heatmap_scale: Heatmap scale for keypoints. Image to heatmap ratio
        flip: Flip the input image horizontally and inference again
        image_ext: Image/keypoints extension
        save_image: Whether to save keypoint maps
        skip_exists: Whether to skip the existing keypoints
    """
    # Convert shape to list if it's a tuple or single value
    if isinstance(shape, int):
        shape = [shape, shape]
    elif isinstance(shape, tuple):
        shape = list(shape)

    # Check required argument
    if output_dir is None:
        raise ValueError("output_dir is required")

    # Create args object to maintain compatibility with existing code
    class Args:
        pass

    args = Args()
    args.pose_checkpoint = pose_checkpoint
    args.det_config = det_config
    args.det_checkpoint = det_checkpoint
    args.images_dir = images_dir
    args.video_path = video_path
    args.fmasks_dir = fmasks_dir
    args.output_dir = output_dir
    args.num_keypoints = num_keypoints
    args.shape = shape
    args.batch_size = batch_size
    args.num_workers = num_workers
    args.gpu_ids = gpu_ids
    args.fp16 = fp16
    args.det_cat_id = det_cat_id
    args.bbox_thr = bbox_thr
    args.nms_thr = nms_thr
    args.kpt_thr = kpt_thr
    args.radius = radius
    args.thickness = thickness
    args.heatmap_scale = heatmap_scale
    args.flip = flip
    args.image_ext = image_ext
    args.save_image = save_image
    args.skip_exists = skip_exists

    if args.det_config is None or args.det_config == "":
        use_det = False
    else:
        use_det = True
        assert has_mmdet, "Please install mmdet to run the demo."
        assert args.det_checkpoint is not None

        from detector_utils import process_images_detector

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

    if args.gpu_ids is None:
        args.gpu_ids = list(range(torch.cuda.device_count()))
    else:
        args.gpu_ids = [int(i) for i in args.gpu_ids.split(",")]
    num_gpus = len(args.gpu_ids)

    if detectors is None:
        if use_det:
            detectors = load_detectors(args.det_config, args.det_checkpoint, args.gpu_ids)
        else:
            detectors = []

    dtype = torch.float16 if args.fp16 else torch.bfloat16
    if pose_estimators is None:
        pose_estimators = load_pose_estimators(args.pose_checkpoint, args.gpu_ids, dtype)

    # hard code the image extension
    if args.images_dir is not None:
        image_paths = sorted(
            glob(f"{args.images_dir}/**/*.jpg", recursive=True) + glob(f"{args.images_dir}/**/*.webp", recursive=True)
        )
        assert args.video_path is None, "video_path and images_dir cannot be used together"

    if args.fmasks_dir is not None and args.fmasks_dir != "None":
        fmask_paths = sorted(glob(f"{args.fmasks_dir}/**/*.png", recursive=True))
    else:
        fmask_paths = None

    scale = args.heatmap_scale
    # do not provide preprocess args for detector as we use mmdet
    if args.video_path is not None:
        inference_dataset = AdhocVideoDataset(args.video_path, fmask_paths)
    else:
        inference_dataset = AdhocImageDataset(image_paths, fmask_paths)

    KPTS_COLORS = COCO_WHOLEBODY_KPTS_COLORS  ## 133 keypoints
    SKELETON_INFO = COCO_WHOLEBODY_SKELETON_INFO

    if args.num_keypoints == 17:
        KPTS_COLORS = COCO_KPTS_COLORS
        SKELETON_INFO = COCO_SKELETON_INFO
    elif args.num_keypoints == 308:
        KPTS_COLORS = GOLIATH_KPTS_COLORS
        SKELETON_INFO = GOLIATH_SKELETON_INFO

    # Per-GPU locks to serialize model inference while keeping I/O parallel
    inference_locks = {gpu_id: threading.Lock() for gpu_id in args.gpu_ids}

    def process_single_batch(idx):
        # get model and data
        gpu_idx = idx % num_gpus
        device = f"cuda:{args.gpu_ids[gpu_idx]}"
        pose_estimator = pose_estimators[gpu_idx]

        image_path, orig_img = inference_dataset[idx]

        # ? we add the cam_label here to fit the easyvolcap data format
        if args.video_path is not None:
            output_path = osp.join(args.output_dir, "/".join(image_path.split("/")[-1:]))
        else:
            output_path = osp.join(args.output_dir, "/".join(image_path.split("/")[-2:]))
        image_ext = osp.splitext(image_path)[1]
        output_json_path = output_path.replace(image_ext, ".json")
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

        # --- GPU lock 1: detection ---
        with inference_locks[args.gpu_ids[gpu_idx]]:
            if use_det:
                detector = detectors[gpu_idx]
                imgs = batch_orig_imgs.copy()[..., [2, 1, 0]]
                bboxes_batch = process_images_detector(args, imgs, detector)
            else:
                bboxes_batch = [[] for _ in range(len(batch_orig_imgs))]

        # --- CPU: bbox defaults + pose preprocessing (no lock) ---
        assert len(bboxes_batch) == valid_images_len

        for i, bboxes in enumerate(bboxes_batch):
            if len(bboxes) == 0:
                bboxes_batch[i] = np.array([[0, 0, orig_img_shape[2], orig_img_shape[1]]])  # orig_img_shape: B H W C → bbox: [x1,y1,x2,y2] = [0,0,W,H]

        img_bbox_map = {}
        for i, bboxes in enumerate(bboxes_batch):
            img_bbox_map[i] = len(bboxes)

        pose_imgs, pose_img_centers, pose_img_scales = [], [], []
        for o_img, bbox_list in zip(batch_orig_imgs, bboxes_batch):
            p_imgs, centers, scales_ = preprocess_pose(
                o_img, bbox_list,
                (input_shape[1], input_shape[2]),
                [123.5, 116.5, 103.5],
                [58.5, 57.0, 57.5],
            )
            pose_imgs.extend(p_imgs)
            pose_img_centers.extend(centers)
            pose_img_scales.extend(scales_)

        local_batch_size = 1  # process one image at a time
        n_pose_batches = (len(pose_imgs) + local_batch_size - 1) // local_batch_size

        # --- GPU lock 2: pose inference ---
        with inference_locks[args.gpu_ids[gpu_idx]]:
            torch.compiler.cudagraph_mark_step_begin()
            pose_results = []
            for i in range(n_pose_batches):
                imgs = torch.stack(pose_imgs[i * local_batch_size : (i + 1) * local_batch_size], dim=0)
                valid_len = len(imgs)
                imgs = fake_pad_images_to_batchsize(imgs, local_batch_size)
                pose_results.extend(
                    batch_inference_topdown(pose_estimator, imgs, dtype=dtype, device=device)[:valid_len]
                )

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

        save_args_list = [
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
        for _args in save_args_list:
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
    fire.Fire(inference_sapiens_pose)
