import os
import os.path as osp
import json
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image


def save_image(path: str, image: Image.Image | torch.Tensor, quality: int = 95):
    if isinstance(image, torch.Tensor):
        image = to_pil_image(image)
    os.makedirs(osp.dirname(path), exist_ok=True)
    image.save(path, quality=quality)


def save_json(path: str, data: dict):
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
