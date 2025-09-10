import os, os.path as osp
import sys
import json
import fire
import numpy as np
from easyvolcap.utils.console_utils import tqdm
from easyvolcap.utils.parallel_utils import parallel_execution


sys.path.append(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
from src.data.utils.camera_parser import parse_cameras
from scripts.preprocess.utils.triang_utils import triangulate_points, project_points


def read_kp2d(path, dtype=np.float64):
    with open(path, "r") as f:
        pose = json.load(f)
    instance = pose["instance_info"][0]
    kp = np.array(instance["keypoints"], dtype=dtype)
    kp_depth = kp_score = None
    if "keypoint_depths" in instance:
        kp_depth = np.array(instance["keypoint_depths"], dtype=dtype)
    if "keypoint_scores" in instance:
        kp_score = np.array(instance["keypoint_scores"], dtype=dtype)

    # re-scale fingers score with hand root score
    kp_score[92:112] *= kp_score[91] ** 2
    kp_score[113:133] *= kp_score[112] ** 2

    return kp, kp_depth, kp_score


def write_kp2d(path, kp, kp_depth=None, kp_score=None):
    instance = {"keypoints": kp.tolist()}
    if kp_depth is not None:
        instance["keypoint_depths"] = kp_depth.tolist()
    if kp_score is not None:
        instance["keypoint_scores"] = kp_score.tolist()
    pose = {"instance_info": [instance]}
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(pose, f, indent=4)


def write_kp3d(path, kp3d, kp3d_reproj):
    instance = {
        "keypoints": kp3d.tolist(),
        "keypoint_reproj": kp3d_reproj.tolist(),
    }
    pose = {"instance_info": [instance]}
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(pose, f, indent=4)


def write_kp3d_pcd(path, kp3d):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(kp3d)
    os.makedirs(osp.dirname(path), exist_ok=True)
    o3d.io.write_point_cloud(path, pcd)


def triangulate_skeleton(
    camera_path: str,
    kp2d_dir: str,
    out_kp3d_dir: str,
    out_pcd_dir: str = None,
    out_kp2d_proj_dir: str = None,
    spa_label_range: list[int] = None,
    spa_label_proj_range: list[int] = None,
    tem_label_range: list[int] = None,
    spa_labels: list[int] = None,
    spa_labels_proj: list[int] = None,
    tem_labels: list[int] = None,
    kp2d_padding: list[int, int] = None,
    intri_scale: float = None,
    skip_exists: bool = False,
    num_workers: int = 1,
    dtype=np.float64,
):
    # parse labels
    if spa_labels is not None:
        if spa_label_range is not None:
            raise ValueError("spa_labels and spa_label_range cannot be specified together")
        spa_labels = [f"{int(i):02d}" for i in spa_labels]
    elif spa_label_range is not None:
        b, e, s = spa_label_range
        spa_labels = [f"{int(i):02d}" for i in range(b, e, s)]
    else:
        spa_labels = sorted(os.listdir(kp2d_dir))

    if spa_labels_proj is not None:
        if spa_label_proj_range is not None:
            raise ValueError("spa_labels_proj and spa_label_proj_range cannot be specified together")
        spa_labels_proj = [f"{int(i):02d}" for i in spa_labels_proj]
    elif spa_label_proj_range is not None:
        b, e, s = spa_label_proj_range
        spa_labels_proj = [f"{int(i):02d}" for i in range(b, e, s)]
    else:
        spa_labels_proj = sorted(os.listdir(kp2d_dir))

    if tem_labels is not None:
        if tem_label_range is not None:
            raise ValueError("tem_labels and tem_label_range cannot be specified together")
        tem_labels = [f"{int(i):06d}" for i in tem_labels]
    elif tem_label_range is not None:
        b, e, s = tem_label_range
        tem_labels = [f"{int(i):06d}" for i in range(b, e, s)]
    else:
        tem_labels = sorted(os.listdir(f"{kp2d_dir}/{spa_labels[0]}"))
        tem_labels = [label.split(".")[0] for label in tem_labels]

    # load cameras
    cams = parse_cameras(camera_path, coord_system="opencv", normalize_scene=False)
    Ks = np.array([cams[label]["K"] for label in spa_labels], dtype=dtype)
    Ts = np.array([np.linalg.inv(cams[label]["pose"]) for label in spa_labels], dtype=dtype)
    Ks_proj = np.array([cams[label]["K"] for label in spa_labels_proj], dtype=dtype)
    Ts_proj = np.array([np.linalg.inv(cams[label]["pose"]) for label in spa_labels_proj], dtype=dtype)

    # scale intrinsics
    if intri_scale is not None:
        Ks = Ks * intri_scale
        Ks[:, -1, -1] = 1.0
        Ks_proj = Ks_proj * intri_scale
        Ks_proj[:, -1, -1] = 1.0

    def triangulate_one_skeleton(tem_label):
        kp2d_paths = [f"{kp2d_dir}/{spa_label}/{tem_label}.json" for spa_label in spa_labels]
        out_kp3d_path = f"{out_kp3d_dir}/{tem_label}.json"
        out_pcd_path = f"{out_pcd_dir}/{tem_label}.ply"
        out_kp2d_proj_paths = [f"{out_kp2d_proj_dir}/{spa_label}/{tem_label}.json" for spa_label in spa_labels_proj]

        if skip_exists and osp.exists(out_kp3d_path):
            try:
                json.load(open(out_kp3d_path, "r"))
                return
            except Exception as e:
                print(f"Error loading {out_kp3d_path}: {e}, skipping...")

        # read 2d keypoints
        kp2d, _, kp2d_score = zip(*[read_kp2d(p, dtype=dtype) for p in kp2d_paths])
        kp2d, kp2d_score = np.stack(kp2d), np.stack(kp2d_score)
        if kp2d_padding is not None:
            kp2d += np.array(kp2d_padding, dtype=dtype)[None]

        # triangulate keypoints
        kp3d, kp3d_reproj, _ = triangulate_points(Ks, Ts, kp2d, kp2d_score)

        # save 3d keypoints
        write_kp3d(out_kp3d_path, kp3d, kp3d_reproj)
        if out_pcd_dir is not None:
            write_kp3d_pcd(out_pcd_path, kp3d)

        # project 2d keypoints
        if out_kp2d_proj_dir is not None:
            kp2d_proj, kp2d_depth_proj, _ = project_points(kp3d, Ks_proj, Ts_proj, kp3d_score=None)
            for i in range(len(out_kp2d_proj_paths)):
                write_kp2d(
                    out_kp2d_proj_paths[i],
                    kp=kp2d_proj[i],
                    kp_depth=kp2d_depth_proj[i],
                    kp_score=None,
                )

    if num_workers > 1:
        parallel_execution(
            tem_labels,
            action=triangulate_one_skeleton,
            print_progress=True,
            desc=f"Triangulating skeletons",
            num_workers=num_workers,
            sequential=False,
        )
    else:
        for tem_label in tqdm(tem_labels, desc="Triangulating skeletons"):
            triangulate_one_skeleton(tem_label)


if __name__ == "__main__":
    # usage:
    # python scripts/preprocess/triangulate_skeleton.py \
    # --camera_path $DATADIR/transforms.json --kp2d_dir $DATADIR/poses_sapiens \
    # --out_kp3d_dir $DATADIR/poses_3d --out_pcd_dir $DATADIR/poses_pcd --out_kp2d_proj_dir $DATADIR/poses_2d
    fire.Fire(triangulate_skeleton)
