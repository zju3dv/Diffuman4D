import os
import fire
import json
import torch
import numpy as np
import open3d as o3d
from PIL import Image
from torchvision.transforms.functional import to_tensor
from easyvolcap.utils.easy_utils import read_camera
from easyvolcap.utils.console_utils import tqdm
from easyvolcap.utils.parallel_utils import parallel_execution


def make_projection_matrix(K, R, t):
    """
    K: (B,3,3), R: (B,3,3), t: (B,3)  (world->cam)
    return P: (B,3,4)
    """
    t = t.view(-1, 3, 1)
    Rt = torch.cat([R, t], dim=-1)  # (B,3,4)
    return K @ Rt


def build_voxel_grid_linspaces(bounds, voxel_size, device):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    xs = torch.arange(xmin, xmax, voxel_size, device=device)
    ys = torch.arange(ymin, ymax, voxel_size, device=device)
    zs = torch.arange(zmin, zmax, voxel_size, device=device)
    return xs, ys, zs, (xs.numel(), ys.numel(), zs.numel())


def load_binary_mask(mask_path: str):
    m = Image.open(mask_path)
    m = to_tensor(m).squeeze(0)
    m = m > 0.5
    return m


def save_pcd(path, pts_torch):
    pts = pts_torch.detach().cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    o3d.io.write_point_cloud(path, pcd)


@torch.no_grad()
def carve_visual_hull(fmasks, Ps, bounds, voxel_size=0.02, batch_size=1e6, min_views=None, device="cuda"):
    """
    Carve the visual hull from the fmasks and Ps.

    Args:
        fmasks: (B,H,W) torch.bool
        Ps: (B,3,4) torch.float32
        bounds: (xmin, xmax, ymin, ymax, zmin, zmax)
        voxel_size: float
        batch_size: int
        min_views: int
    Returns:
        (M,3) torch.float32
        M: number of points in the visual hull
    """

    fmasks = fmasks.to(device=device)
    Ps = Ps.to(device=device)

    B, H, W = fmasks.shape
    xs, ys, zs, grid_shape = build_voxel_grid_linspaces(bounds, voxel_size, device)

    nx, ny, nz = grid_shape
    N = nx * ny * nz
    batch_size = int(batch_size)
    P = Ps

    kept_points = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        idx = torch.arange(start, end, device=device, dtype=torch.long)  # (M,)

        iz = idx % nz
        iy = (idx // nz) % ny
        ix = idx // (ny * nz)

        # voxel center points (M,3)
        X = torch.stack([xs[ix], ys[iy], zs[iz]], dim=-1).to(P.dtype)
        ones = torch.ones((X.shape[0], 1), device=device, dtype=X.dtype)
        Xh = torch.cat([X, ones], dim=-1)  # (M,4)

        # projection: x = P @ Xh
        x = torch.matmul(P, Xh.t()).transpose(1, 2)  # (B,M,3)

        z = x[..., 2]  # (B,M)
        uv = x[..., :2] / (z.unsqueeze(-1).clamp_min(1e-8))  # (B,M,2)

        u = torch.round(uv[..., 0]).to(torch.long)  # (B,M)
        v = torch.round(uv[..., 1]).to(torch.long)  # (B,M)

        valid = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)  # (B,M)

        # sample foreground: default invalid projection is False (will be carved away)
        inside = torch.zeros_like(valid, dtype=torch.bool)  # (B,M)
        if valid.any():
            # advanced indexing: mask[b, v, u]
            b_idx = torch.arange(B, device=device).view(B, 1).expand_as(u)  # (B,M)
            inside_valid = fmasks[b_idx[valid], v[valid], u[valid]]
            inside[valid] = inside_valid

        if min_views is None:
            # must all views be inside
            keep = inside.all(dim=0)  # (M,)
        else:
            # at least k views inside
            keep = inside.sum(dim=0) >= int(min_views)

        if keep.any():
            kept_points.append(X[keep])

    if len(kept_points) == 0:
        return torch.empty((0, 3), device=device, dtype=P.dtype)

    hull_pts = torch.cat(kept_points, dim=0)  # (M,3)
    return hull_pts


def main(
    fmasks_dir: str = "data/diffuman4d_example/0013_01/fmasks",
    cameras_path: str = "data/diffuman4d_example/0013_01/transforms.json",
    out_vhull_dir: str = "data/diffuman4d_example/0013_01/surfs",
    camera_range: tuple[int] = (0, None, 1),
    frame_range: tuple[int] = (0, None, 1),
    bounds: tuple[float] = (-3.0, 3.0, -3.0, 3.0, -3.0, 3.0),
    voxel_size: float = 0.025,
    batch_size: int = 1e6,
    min_views: int = None,
    save_nerfstudio_pcd: bool = True,
    device: str = "cuda",
):
    """
    Carve the visual hull from foreground masks.
    The input data should be organized as EasyVolcap format.

    Args:
        bounds: tuple[float]: try enlarge it if the result is empty
        voxel_size: float: smaller voxel size for more details
        batch_size: int: decrease it if OOM
        min_views: int: at least k views inside, all views by default
    """

    cam_labels = sorted(os.listdir(fmasks_dir))
    frm_labels = sorted(os.listdir(os.path.join(fmasks_dir, cam_labels[0])))
    frm_labels = [os.path.splitext(frm_label)[0] for frm_label in frm_labels]

    b, e, s = camera_range
    cam_labels = cam_labels[b:e:s]
    b, e, s = frame_range
    frm_labels = frm_labels[b:e:s]

    if cameras_path.endswith(".json"):
        # load nerfstudio cameras
        tf = json.load(open(cameras_path))["frames"]
        Ks = np.stack(
            [[[frame["fl_x"], 0, frame["cx"]], [0, frame["fl_y"], frame["cy"]], [0, 0, 1]] for frame in tf],
            axis=0,
        )
        c2ws = np.stack([frame["transform_matrix"] for frame in tf], axis=0)
        c2ws[:, :3, 1:3] *= -1  # opengl to opencv
        w2cs = np.linalg.inv(c2ws)
        Rs = w2cs[:, :3, :3]
        Ts = w2cs[:, :3, 3:4]
        P = make_projection_matrix(torch.from_numpy(Ks), torch.from_numpy(Rs), torch.from_numpy(Ts))
    else:
        # load easyvolcap cameras
        cameras = read_camera(cameras_path)
        Ks = np.stack([cameras[cam_label]["K"] for cam_label in cam_labels], axis=0)
        Rs = np.stack([cameras[cam_label]["R"] for cam_label in cam_labels], axis=0)
        Ts = np.stack([cameras[cam_label]["T"] for cam_label in cam_labels], axis=0)
        P = make_projection_matrix(torch.from_numpy(Ks), torch.from_numpy(Rs), torch.from_numpy(Ts))

    fmask_paths = [
        os.path.join(fmasks_dir, cam_label, f"{frm_label}.png") for frm_label in frm_labels for cam_label in cam_labels
    ]
    fmasks = parallel_execution(
        fmask_paths, action=load_binary_mask, num_workers=16, print_progress=True, desc="Loading foreground masks"
    )
    H, W = fmasks[0].shape
    fmasks = torch.stack(fmasks, dim=0).reshape(len(frm_labels), len(cam_labels), H, W)

    for i, frm_label in tqdm(enumerate(frm_labels), total=len(frm_labels), desc="Carving visual hulls"):
        hull_pts = carve_visual_hull(
            fmasks[i], P, bounds, voxel_size=voxel_size, batch_size=batch_size, min_views=min_views, device=device
        )

        save_pcd(os.path.join(out_vhull_dir, f"{frm_label}.ply"), hull_pts)

        if i == 0 and save_nerfstudio_pcd:
            out_pcd_dir = os.path.dirname(cameras_path) if cameras_path.endswith(".json") else cameras_path
            save_pcd(f"{out_pcd_dir}/sparse_pcd.ply", hull_pts)


if __name__ == "__main__":
    fire.Fire(main)
