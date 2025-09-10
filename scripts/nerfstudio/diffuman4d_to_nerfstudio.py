import os
import os.path as osp
import json
import fire
import shutil

from copy import deepcopy
from src.utils import RankedLogger
from scripts.preprocess.remove_background import remove_background

log = RankedLogger(__name__, rank_zero_only=True)


def diffuman4d_to_nerfstudio(data_dir: str, result_dir: str, input_cameras: list[str] = None):
    # copy nerfstudio cameras
    cameras_path = f"{data_dir}/transforms.json"
    cameras = json.load(open(cameras_path))

    if input_cameras is not None:
        cameras_input = deepcopy(cameras)
        cameras_input["frames"] = []

    for frame in cameras["frames"]:
        ext = osp.splitext(frame["file_path"])[1]
        frame["file_path"] = frame["file_path"].replace(ext, ".png").replace("images/", "images_alpha/")
        # record input cameras
        if input_cameras is not None and frame["camera_label"] in input_cameras:
            cameras_input["frames"].append(frame)

    os.makedirs(result_dir, exist_ok=True)
    with open(f"{result_dir}/transforms.json", "w") as f:
        json.dump(cameras, f, indent=4)
    with open(f"{result_dir}/transforms_input.json", "w") as f:
        json.dump(cameras_input, f, indent=4)
    log.info(f"Saved nerfstudio cameras to {result_dir}/transforms.json and {result_dir}/transforms_input.json")

    # copy point cloud
    shutil.copy(f"{data_dir}/sparse_pcd.ply", f"{result_dir}/sparse_pcd.ply")
    log.info(f"Saved point cloud to {result_dir}/sparse_pcd.ply")

    # predict foreground masks
    remove_background(
        images_dir=f"{result_dir}/images",
        out_fmasks_dir=f"{result_dir}/fmasks",
        out_images_alpha_dir=f"{result_dir}/images_alpha",
        model_name="ZhengPeng7/BiRefNet",
        image_ext=".jpg",
        mask_ext=".png",
        rotate_clockwise=0,
        batch_size=4,  # decrease it if OOM
    )
    log.info(f"Saved foreground masks to {result_dir}/fmasks")


if __name__ == "__main__":
    fire.Fire(diffuman4d_to_nerfstudio)
