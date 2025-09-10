import torch
from PIL import Image
from einops import rearrange
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid, save_image
from torchvision.transforms import InterpolationMode


def denorm_vae_tensor(img: torch.Tensor) -> torch.Tensor:
    return img * 0.5 + 0.5  # [1, -1] -> [0, 1]


def norm_vae_tensor(img: torch.Tensor) -> torch.Tensor:
    return img * 2.0 - 1.0  # [0, 1] -> [-1, 1]


def vae_tensor_to_pil(img: torch.Tensor) -> Image.Image:
    return TF.to_pil_image(denorm_vae_tensor(img))


def pil_to_vae_tensor(img: Image.Image) -> torch.Tensor:
    return norm_vae_tensor(TF.to_tensor(img))


def apply_fmask(
    image: torch.Tensor, fmask: torch.Tensor, background_color: str = "white", vae_normalized: bool = False
) -> torch.Tensor:
    """
    Apply the fmask to the image.

    Args:
        image: [3, h, w] in [0, 1]
        fmask: [1, h, w] in [0, 1]
        background_color: "white", "black", "random"

    Returns:
        image: [3, h, w] in [0, 1]
    """
    if vae_normalized:
        image = denorm_vae_tensor(image)
        fmask = denorm_vae_tensor(fmask)

    bmask = 1.0 - fmask
    if background_color == "white":
        background = bmask * 1.0
    elif background_color == "black":
        background = bmask * 0.0
    elif background_color == "random":
        background = torch.normal(mean=0, std=0.3, size=(3,)).clamp_(0.0, 1.0)
        background = background[:, None, None] * bmask
    else:
        raise ValueError(f"Invalid background color: {background_color}")

    image = image * fmask + background

    if vae_normalized:
        image = norm_vae_tensor(image)

    return image


def restore_cropped_image(
    image: Image.Image, crop_param: tuple[int, ...], ori_size: tuple[int, int] = None, background_color: str = "white"
) -> Image.Image:
    # the image is first cropped with (ct, cl, ch, cw) and then resized to (h, w)
    if len(crop_param) == 4:
        ct, cl, ch, cw = crop_param
        w, h = image.size
    elif len(crop_param) == 6:
        ct, cl, ch, cw, h, w = crop_param
    else:
        raise ValueError(f"Invalid crop_param: {crop_param}")

    # 1. revert the resizing
    image = TF.resize(image, (ch, cw), interpolation=InterpolationMode.BICUBIC)
    image = TF.to_tensor(image)

    # 2. revert the cropping (init a big canvas for padding)
    canvas = torch.zeros((image.shape[0], h * 2, w * 2))
    if background_color == "white":
        canvas[...] = 1.0

    left = w // 2 + cl
    top = h // 2 + ct
    right = left + cw
    bottom = top + ch
    canvas[:, top:bottom, left:right] = image

    # crop the canvas
    restored_image = canvas[:, h // 2 : h * 3 // 2, w // 2 : w * 3 // 2]

    restored_image = TF.to_pil_image(restored_image)
    return restored_image
