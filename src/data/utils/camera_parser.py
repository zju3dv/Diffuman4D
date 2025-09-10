import os
import os.path as osp
import json
import torch


def normalize_poses(poses: torch.Tensor, center: torch.Tensor = None, scale: float = None):
    def calc_scene_scale(points: torch.Tensor):
        min_bound = torch.min(points, dim=0).values
        max_bound = torch.max(points, dim=0).values
        center = (min_bound + max_bound) / 2
        scale = 1 / torch.linalg.norm(max_bound - min_bound)
        return center, scale

    if center is None or scale is None:
        center, scale = calc_scene_scale(poses[:, :3, 3])
    poses[:, :3, 3] = (poses[:, :3, 3] - center) * scale  # in-place update


def parse_cameras(camera_path: str, coord_system: str = "opencv", normalize_scene: bool = True) -> dict[str, dict]:
    """
    Parse nerfstudio /easyvolcap cameras to opencv or opengl coordinate system.
    """

    Ks, hws, poses, labels = [], [], [], []

    # read nerfstudio camera
    if camera_path.endswith(".json"):
        with open(camera_path, "r") as f:
            tfs = json.load(f)
        cams = tfs["frames"]

        # Load camera intrinsic and extrinsic
        for cam in cams:
            if "fl_x" in cam and "fl_y" in cam and "cx" in cam and "cy" in cam:
                fx, fy, cx, cy = cam["fl_x"], cam["fl_y"], cam["cx"], cam["cy"]
            else:
                fx, fy, cx, cy = tfs["fl_x"], tfs["fl_y"], tfs["cx"], tfs["cy"]
            Ks.append(torch.tensor([fx, 0, cx, 0, fy, cy, 0, 0, 1]).reshape(3, 3))
            hws.append((cam["h"], cam["w"]))

            pose = torch.tensor(cam["transform_matrix"])
            pose[:3, 1:3] *= -1  # convert to opencv as default
            poses.append(pose)

            labels.append(cam["camera_label"])

    # read easyvolcap camera
    elif osp.isdir(camera_path) or camera_path.endswith(".yml"):
        from easyvolcap.utils.easy_utils import read_camera

        cams = read_camera(camera_path)

        # Load camera intrinsic and extrinsic
        for label, cam in cams.items():
            K = cam["K"]
            h, w = cam["H"], cam["W"]
            if h < 0 or w < 0:
                raise ValueError(f"Invalid camera height or width: {h}, {w}")
            Ks.append(torch.from_numpy(K).reshape(3, 3))
            hws.append((h, w))

            w2c = torch.eye(4)
            w2c[:3, :] = torch.from_numpy(cam["RT"])
            c2w = torch.linalg.inv(w2c)
            poses.append(c2w)

            labels.append(label)

    # Coordinate system conversion
    poses = torch.stack(poses)
    if coord_system == "opengl":
        # convert from opencv to opengl
        poses[:, 0:3, 1:3] *= -1

    # Normalize the poses
    if normalize_scene:
        norm_json = f"{camera_path}/scene_norm.json"
        if os.path.isfile(norm_json):
            norm_data = json.load(open(norm_json))
            center = torch.tensor(norm_data["center"])
            scale = norm_data["scale"]
        else:
            center = scale = None
        normalize_poses(poses=poses, center=center, scale=scale)

    # Build the camera dict
    data = {}
    for label, K, hw, pose in zip(labels, Ks, hws, poses):
        data[label] = {"K": K, "pose": pose, "height": hw[0], "width": hw[1]}
    return data
