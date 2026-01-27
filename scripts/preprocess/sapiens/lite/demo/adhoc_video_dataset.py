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


def read_video(path: str, frame_range: list[int, int, int] = None) -> torch.Tensor:
    from video_reader import PyVideoReader

    if frame_range is None or frame_range[1] is None:
        # None case
        video = PyVideoReader(path).decode()

        if frame_range is not None and frame_range[1] is None:
            # [begine, None, step] case
            video = video[frame_range[0] : frame_range[1] : frame_range[2]]
    else:
        # [begin, end, step] case
        video = PyVideoReader(path).decode(start_frame=frame_range[0], end_frame=frame_range[1])
        video = video[:: frame_range[2]]

    video = torch.from_numpy(video).to(dtype=torch.float32) / 255.0
    video = video.permute(0, 3, 1, 2)
    return video


class AdhocVideoDataset(torch.utils.data.Dataset):
    def __init__(self, video_path, fmask_paths=None, shape=None):
        self.video_path = video_path
        self.fmask_paths = fmask_paths

        self.video = read_video(video_path)

        if self.fmask_paths is not None and len(self.fmask_paths) != len(self.video):
            raise ValueError("fmask_paths and video must have the same length")

        if shape is None:
            H, W = self.video.shape[-2:]
            height, width = 1024, int(round(W / H * 1024))
            self.shape = (height, width)
        else:
            assert len(shape) == 2
            self.shape = shape

        # (H, W) resize to (1024, W / H * 1024)
        self.tranform_image = transforms.Compose(
            [
                transforms.Resize(self.shape, interpolation=transforms.InterpolationMode.BICUBIC),
            ]
        )

    def __len__(self):
        return len(self.video)

    def __getitem__(self, idx):
        image = TF.to_pil_image(self.video[idx])
        if self.fmask_paths is not None:
            fmask_path = self.fmask_paths[idx]
            fmask = Image.open(fmask_path)
            bg = Image.new(image.mode, image.size, (0, 0, 0))
            image = Image.composite(image, bg, fmask)

        image = self.tranform_image(image)
        image = np.array(image)
        image_path = self.video_path.replace(".mp4", f"/{idx:06d}.webp")
        return image_path, image
