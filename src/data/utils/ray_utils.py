# borrowed from https://github.com/zju3dv/EasyVolcap/blob/main/easyvolcap/utils/ray_utils.py
import torch


@torch.jit.script
def normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # channel last: normalization
    return x / (torch.norm(x, dim=-1, keepdim=True) + eps)


def get_rays(
    H: int,
    W: int,
    K: torch.Tensor,
    R: torch.Tensor,
    T: torch.Tensor,
    is_inv_K: bool = False,
    z_depth: bool = False,
    correct_pix: bool = True,
    ret_coord: bool = False,
):
    # calculate the world coodinates of pixels
    i, j = torch.meshgrid(
        torch.arange(H, dtype=R.dtype, device=R.device),
        torch.arange(W, dtype=R.dtype, device=R.device),
        indexing="ij",
    )
    bss = K.shape[:-2]
    for _ in range(len(bss)):
        i, j = i[None], j[None]
    i, j = i.expand(bss + i.shape[len(bss) :]), j.expand(bss + j.shape[len(bss) :])
    # 0->H, 0->W
    return get_rays_from_ij(i, j, K, R, T, is_inv_K, z_depth, correct_pix, ret_coord)


def get_rays_from_ij(
    i: torch.Tensor,
    j: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    T: torch.Tensor,
    is_inv_K: bool = False,
    use_z_depth: bool = False,
    correct_pix: bool = True,
    ret_coord: bool = False,
):
    # i: B, P or B, H, W or P or H, W
    # j: B, P or B, H, W or P or H, W
    # K: B, 3, 3
    # R: B, 3, 3
    # T: B, 3, 1
    nb_dim = len(K.shape[:-2])  # number of batch dimensions
    np_dim = len(i.shape[nb_dim:])  # number of points dimensions
    if not is_inv_K:
        invK = torch.inverse(K.float()).type(K.dtype)
    else:
        invK = K
    ray_o = -R.mT @ T  # B, 3, 1

    # Prepare the shapes
    for _ in range(np_dim):
        invK = invK.unsqueeze(-3)
    invK = invK.expand(i.shape + (3, 3))
    for _ in range(np_dim):
        R = R.unsqueeze(-3)
    R = R.expand(i.shape + (3, 3))
    for _ in range(np_dim):
        T = T.unsqueeze(-3)
    T = T.expand(i.shape + (3, 1))
    for _ in range(np_dim):
        ray_o = ray_o.unsqueeze(-3)
    ray_o = ray_o.expand(i.shape + (3, 1))

    # Pixel center correction
    if correct_pix:
        i, j = i + 0.5, j + 0.5
    else:
        i, j = i.float(), j.float()

    # 0->H, 0->W
    # int -> float; # B, H, W, 3, 1 or B, P, 3, 1 or P, 3, 1 or H, W, 3, 1
    xy1 = torch.stack([j, i, torch.ones_like(i)], dim=-1)[..., None]
    pixel_camera = invK @ xy1  # B, H, W, 3, 1 or B, P, 3, 1
    pixel_world = R.mT @ (pixel_camera - T)  # B, P, 3, 1

    # Calculate the ray direction
    pixel_world = pixel_world[..., 0]
    ray_o = ray_o[..., 0]
    ray_d = pixel_world - ray_o  # use pixel_world depth as is (no curving)
    if not use_z_depth:
        ray_d = normalize(ray_d)  # B, P, 3, 1

    if not ret_coord:
        return ray_o, ray_d
    elif correct_pix:
        return ray_o, ray_d, (torch.stack([i, j], dim=-1) - 0.5).long()  # B, P, 2
    else:
        return ray_o, ray_d, torch.stack([i, j], dim=-1).long()  # B, P, 2


def calc_plucker_embeds(h: int, w: int, K: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
    extrinsic = torch.inverse(pose)
    R = extrinsic[:, :3, :3]  # (B, 3, 3)
    T = extrinsic[:, :3, 3:]  # (B, 3, 1)
    # Compute the ray origins, directions
    ray_o, ray_d = get_rays(h, w, K, R, T)  # B, H, W, 3

    # Compute the plucker embedding
    plucker_normal = torch.cross(ray_o, ray_d, dim=-1)
    plucker_embeds = torch.cat([ray_d, plucker_normal], dim=-1)
    plucker_embeds = plucker_embeds.permute(0, 3, 1, 2)  # B, 6, H, W
    return plucker_embeds


def calc_relative_poses(poses: torch.Tensor) -> torch.Tensor:
    ref_pose = poses[0]
    ref_pose_inv = torch.inverse(ref_pose)
    rel_poses = torch.matmul(ref_pose_inv, poses)
    return rel_poses
