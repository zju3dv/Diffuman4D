import os
import os.path as osp
import fire
import json
import copy
import torch
import cv2
import numpy as np
from PIL import Image
from glob import glob

from easyvolcap.utils.console_utils import log
from easyvolcap.utils.parallel_utils import parallel_execution
from easyvolcap.utils.easy_utils import read_camera, write_camera
from easyvolcap.utils.undist_utils import colmap_undistort

from scripts.download.utils.SMCReader import SMCReader


def calc_unified_cameras(cams, image_size=1024):
    def transform_resize(K, h, w, tar_f):
        K = copy.deepcopy(K)
        scale_w = tar_f / K[0, 0]
        scale_h = tar_f / K[1, 1]
        tar_w = int(round(w * scale_w))
        tar_h = int(round(h * scale_h))
        K[0, 0] *= scale_w
        K[0, 2] *= scale_w
        K[1, 1] *= scale_h
        K[1, 2] *= scale_h
        return K, tar_h, tar_w

    def transform_crop(K, h, w, tar_h, tar_w):
        K = copy.deepcopy(K)
        cx, cy = K[0, 2], K[1, 2]
        # move principal point to the center
        left = int(round(cx - tar_w // 2))
        right = int(round(cx + tar_w // 2))
        top = int(round(cy - tar_h // 2))
        bottom = int(round(cy + tar_h // 2))
        assert left >= 0 and right <= w and top >= 0 and bottom <= h
        K[0, 2], K[1, 2] = tar_w / 2, tar_h / 2
        return K, left, top, right, bottom

    cams = copy.deepcopy(cams)
    cam_labels = sorted(list(cams.keys()))

    # Processing cameras
    resized_whs = []
    cropped_ltrbs = []

    for cam_id, cam_label in enumerate(cam_labels):
        # read camera
        cam = cams[cam_label]
        K = cam["K"]
        h, w = cam["H"], cam["W"]

        if 0 <= cam_id <= 47:
            tar_f = 2496 * (image_size / 1920)
            tar_h = image_size
            tar_w = image_size
        elif 48 <= cam_id <= 59:
            tar_f = 3648 * (image_size / 1920)
            tar_h = int(2880 * (image_size / 1920))
            tar_w = int(2880 * (image_size / 1920))
        else:
            raise ValueError(f"Unknown camera id: {cam_id}")

        # calculate resized and cropped intrinsics
        resized_K, resized_h, resized_w = transform_resize(K, h, w, tar_f)
        cropped_K, left, top, right, bottom = transform_crop(resized_K, resized_h, resized_w, tar_h, tar_w)

        assert (
            cropped_K[0, 0] - tar_f < 1e-6
            and cropped_K[1, 1] - tar_f < 1e-6
            and cropped_K[0, 2] == tar_w / 2
            and cropped_K[1, 2] == tar_h / 2
        )
        # update the camera parameters
        cam["K"] = cropped_K
        cam["H"] = tar_h
        cam["W"] = tar_w
        cam["resized_wh"] = (resized_w, resized_h)
        cam["cropped_ltrb"] = (left, top, right, bottom)

    return cams


def calib_undist_image(image, K, D, ccm_data, resized_wh, cropped_ltrb):
    def calib_color(img_ori, bgr_sol):
        img_ori = img_ori[None, ...]
        bgr_sol = bgr_sol[None, ...]

        # Rearrange bgr_sol indices to match channel order
        bgr_sol_perm = bgr_sol[:, [2, 1, 0], :]
        # Construct X matrix with shape [B, H, W, 3, 3]
        X = torch.stack([img_ori**2, img_ori, torch.ones_like(img_ori)], dim=-1)
        # Compute the result using broadcasting and sum over the last dimension
        rs_img = (X * bgr_sol_perm[:, None, None, :, :]).sum(dim=-1)
        return rs_img[0].clamp(0, 255)

    image = torch.from_numpy(image.copy()).float().to("cuda")

    # correct image color
    ccm_data = torch.from_numpy(ccm_data).float().to("cuda")
    image = calib_color(image, ccm_data)

    # undistort image
    image = image.to("cuda", non_blocking=True)
    K = torch.from_numpy(K).float().to("cuda", non_blocking=True)
    D = torch.from_numpy(D).float().to("cuda", non_blocking=True)
    image = colmap_undistort(image, K, D)[0]
    image = image.cpu().numpy().astype(np.uint8)

    # resize and crop image
    image = cv2.resize(image, resized_wh, interpolation=cv2.INTER_AREA)
    left, top, right, bottom = cropped_ltrb
    image = image[top:bottom, left:right]

    return image


def extract_images(
    smc_path="DNA-Rendering/dna_rendering_release_data/Part 2/dna_rendering_part2_main/0007_01.smc",
    cameras_dir="DNA-Rendering/dna_rendering_processed/0007_01/cameras",
    out_images_dir="DNA-Rendering/dna_rendering_processed/0007_01/images",
    image_ext=".webp",
    image_quality=85,
    skip_exists=True,
    num_workers=12,
):
    rd = SMCReader(smc_path)
    available_keys = rd.get_available_keys()
    log(f"Loaded '{smc_path}' with available keys: {available_keys}")

    # cameras for color correction
    ccm_dir = f"{cameras_dir}/ccm"
    ccm_cameras = read_camera(ccm_dir)
    # cameras for undistortion
    colmap_dir = f"{cameras_dir}/colmap/sparse/0"
    dist_cameras = read_camera(colmap_dir)
    colmap_dense_dir = f"{cameras_dir}/colmap/dense/sparse/0"
    undist_cameras = read_camera(colmap_dense_dir)

    # unify camera intrinsics
    unified_cameras = calc_unified_cameras(undist_cameras)

    # extract images
    def extract_one_image(key):
        cam_group, cam_id, frame_id = key
        cam_label = f"{int(cam_id):02d}"
        frame_label = f"{int(frame_id):06d}"
        image_path = f"{out_images_dir}/{cam_label}/{frame_label}{image_ext}"

        # strcitly verify the file
        if skip_exists and osp.exists(image_path):
            try:
                Image.open(image_path).verify()
                return
            except Exception as e:
                os.remove(image_path)

        image = rd.get_img(cam_group, cam_id, "color", frame_id)
        image = np.flip(image, axis=-1)

        image = calib_undist_image(
            image,
            K=dist_cameras[cam_label].K,
            D=dist_cameras[cam_label].D,
            ccm_data=ccm_cameras[cam_label].ccm,
            resized_wh=unified_cameras[cam_label].resized_wh,
            cropped_ltrb=unified_cameras[cam_label].cropped_ltrb,
        )

        os.makedirs(osp.dirname(image_path), exist_ok=True)
        Image.fromarray(image).save(image_path, quality=image_quality)

    # get camera and frame ids
    cam_group = "Camera_5mp"
    camera_ids = list(range(48))
    frame_ids = sorted([int(l) for l in rd.smc["Camera_5mp"]["0"]["color"].keys()])
    keys = [(cam_group, camera_id, frame_id) for camera_id in camera_ids for frame_id in frame_ids]

    parallel_execution(
        keys,
        action=extract_one_image,
        sequential=False,
        num_workers=num_workers,
        print_progress=True,
        desc=f"Extracting RGB images from '{smc_path}'",
    )

    rd.release()


def extract_dnar_images(
    raw_root: str = "./data/dna_rendering_release_data",
    processed_root: str = "./data/dna_rendering_processed",
    scenes: list[str] = None,
):
    """
    Extract RGB images from official DNA-Rendering dataset and save to:
    - scene/images/<cam>/<frame>.webp

    Args:
        raw_root: the root directory of the raw DNA-Rendering dataset downloaded from https://dna-rendering.github.io/
        processed_root: the root directory of the re-annotated labels for the DNA-Rendering dataset downloaded from https://huggingface.co/datasets/krahets/dna_rendering_processed
        scenes: the scenes to extract. If None, all scenes in processed_root will be processed.
    """

    def get_scene_parts(raw_root):
        # please download Part_*_file_gid.json from official DNA-Rendering website
        gid_paths = sorted(glob(f"{raw_root}/Part*/Part_*_file_gid.json"))
        if len(gid_paths) == 0:
            raise ValueError(
                f"No Part_*_file_gid.json found in {raw_root}. Please download it from https://dna-rendering.github.io/"
            )
        filepaths_gids = {}
        for gid_path in gid_paths:
            filepaths_gids.update(json.load(open(gid_path)))

        scene_parts = {}
        for file_path in filepaths_gids.keys():
            scene_label = osp.basename(file_path).split(".")[0]
            part = file_path.split("/")[0]
            scene_parts[scene_label] = part

        return scene_parts

    if scenes is None:
        scenes = sorted(os.listdir(processed_root))
        log(f"Found {len(scenes)} scenes in {processed_root}.")

    scene_parts = get_scene_parts(raw_root)
    parts = [scene_parts[scene] for scene in scenes]

    for i, (scene, part) in enumerate(zip(scenes, parts)):
        log(f"Start extracting images for {scene} in {part} ({i+1}/{len(scenes)}).")
        extract_images(
            smc_path=f"{raw_root}/{part}/dna_rendering_{part.lower().replace(' ', '')}_main/{scene}.smc",
            cameras_dir=f"{processed_root}/{scene}/cameras",
            out_images_dir=f"{processed_root}/{scene}/images",
        )


if __name__ == "__main__":
    # This script is for extracting the RGB images from the raw DNA-Rendering dataset (https://dna-rendering.github.io/index.html).
    # requirements: pip install -U huggingface_hub datasets pyarrow pandas && pip install git+https://github.com/zju3dv/EasyVolcap.git --no-deps
    # usage: python scripts/download/extract_dnar_images.py -h
    fire.Fire(extract_dnar_images)
