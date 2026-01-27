# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image


class AdhocImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, fmask_paths=None, shape=None):
        self.image_paths = image_paths
        self.fmask_paths = fmask_paths
        if self.fmask_paths is not None and len(self.fmask_paths) != len(self.image_paths):
            raise ValueError("fmask_paths and image_paths must have the same length")
        if shape:
            assert len(shape) == 2

        if self.shape is None:
            image = Image.open(image_paths[0])
            W, H = image.size
            height, width = 1024, int(round(W / H * 1024))
            self.shape = (height, width)
        else:
            self.shape = shape

        # (H, W) resize to (1024, W / H * 1024) crop to (h, w)
        self.tranform_image = transforms.Compose(
            [
                transforms.Resize(self.shape[0], interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(self.shape),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        if self.fmask_paths is not None:
            fmask_path = self.fmask_paths[idx]
            fmask = Image.open(fmask_path)
            bg = Image.new(image.mode, image.size, (0, 0, 0))
            image = Image.composite(image, bg, fmask)

        image = self.tranform_image(image)
        image = np.array(image)
        return image_path, image
