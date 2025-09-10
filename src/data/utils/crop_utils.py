import math
import torch
from PIL import Image
import torchvision.transforms.functional as TF


def mask_to_bbox(fmask: Image.Image) -> tuple[int, int, int, int] | None:
    # get the bounding box
    if isinstance(fmask, Image.Image):
        fmask = TF.to_tensor(fmask)
    if fmask.dim() == 3:
        fmask = fmask.mean(dim=0)
    rows = torch.any(fmask, dim=1).nonzero(as_tuple=True)[0]
    cols = torch.any(fmask, dim=0).nonzero(as_tuple=True)[0]
    if rows.numel() == 0 or cols.numel() == 0:
        return None

    xmin, ymin, xmax, ymax = (cols[0].item() - 1, rows[0].item() - 1, cols[-1].item() + 1, rows[-1].item() + 1)
    return xmin, ymin, xmax, ymax


def mask_crop_aspect_ratio(
    fmask: Image.Image,
    aspect_ratio: float = 1.0,
    center_principal_point: bool = False,
    min_crop_ratio: float = 0.7,
    crop_padding_range: tuple[int, int] = (0, 1),
) -> tuple[int, ...]:
    """
    Crop the image with the mask according to the aspect ratio.
    """
    w, h = fmask.size

    # 1. get the bounding box of the mask
    xmin, ymin, xmax, ymax = mask_to_bbox(fmask)

    # 2. get the bounding box according to the aspect ratio
    if center_principal_point:
        # principal point
        xctr, yctr = w / 2, h / 2
    else:
        # center of the bounding box
        xctr, yctr = (xmin + xmax) / 2, (ymin + ymax) / 2
    height = 2 * max(yctr - ymin, ymax - yctr, (xctr - xmin) * aspect_ratio, (xmax - xctr) * aspect_ratio)

    # ensure the cropped image is not too small
    min_height = min_crop_ratio * h
    height = max(height, min_height)

    # ensure the crop is inside the image to avoid invalid pixels
    if center_principal_point:
        max_height = 2 * min(h - yctr, yctr)
        max_width = 2 * min(w - xctr, xctr)
        height = min(height, max_height, max_width * aspect_ratio)
    width = int(height / aspect_ratio)
    xmin, ymin, xmax, ymax = (xctr - width / 2, yctr - height / 2, xctr + width / 2, yctr + height / 2)

    # 3. add random padding for data augmentation
    padding = torch.randint(*crop_padding_range, (1,)).item()
    # ensure the padding is inside the image
    padding = max(min(padding, xmin, ymin, w - xmax, h - ymax), 0)
    xmin, ymin, xmax, ymax = (xmin - padding, ymin - padding, xmax + padding, ymax + padding)
    xmin, ymin, xmax, ymax = math.floor(xmin), math.floor(ymin), math.ceil(xmax), math.ceil(ymax)

    top, left, height, width = ymin, xmin, ymax - ymin, xmax - xmin
    return [top, left, height, width, h, w]


def skeleton_to_mask(skeleton: Image.Image, padding_ratio: float = 0.03) -> Image.Image:
    w, h = skeleton.size
    py, px = int(h * padding_ratio), int(w * padding_ratio)
    pt = int(py * 3)

    skeleton = TF.to_tensor(skeleton)
    fmask = skeleton.mean(dim=0)
    xmin, ymin, xmax, ymax = mask_to_bbox(fmask)
    xmin, ymin, xmax, ymax = (max(xmin - px, 0), max(ymin - pt, 0), min(xmax + px, w), min(ymax + py, h))
    fmask[ymin:ymax, xmin:xmax] = 1.0
    fmask = TF.to_pil_image(fmask)
    return fmask
