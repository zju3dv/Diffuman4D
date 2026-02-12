# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import torch
import numpy as np
from PIL import Image


class AdhocImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, fmask_paths=None):
        self.image_paths = image_paths
        self.fmask_paths = fmask_paths
        if self.fmask_paths is not None and len(self.fmask_paths) != len(self.image_paths):
            raise ValueError("fmask_paths and image_paths must have the same length")

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

        orig_image = np.array(image)[..., [2, 1, 0]]  # RGB to BGR
        return image_path, orig_image
