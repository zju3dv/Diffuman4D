# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
from torchcodec.decoders import SimpleVideoDecoder  # torchcodec==0.0.3 for torch==2.4.1


def read_video(path: str, frame_range: list[int, int, int] | None = None) -> torch.Tensor:
    decoder = SimpleVideoDecoder(path)

    if frame_range is not None:
        b, e, s = frame_range
    else:
        b, e, s = 0, len(decoder), 1
    frames = decoder.get_frames_at(start=b, stop=e, step=s)

    # FrameBatch.data: uint8 tensor [F, C, H, W]
    video = frames.data.to(dtype=torch.float32).div_(255.0)

    return video


class AdhocVideoDataset(torch.utils.data.Dataset):
    def __init__(self, video_path, fmask_paths=None):
        self.video_path = video_path
        self.fmask_paths = fmask_paths

        self.video = read_video(video_path)

        if self.fmask_paths is not None and len(self.fmask_paths) != len(self.video):
            raise ValueError("fmask_paths and video must have the same length")

    def __len__(self):
        return len(self.video)

    def __getitem__(self, idx):
        image = TF.to_pil_image(self.video[idx])
        if self.fmask_paths is not None:
            fmask_path = self.fmask_paths[idx]
            fmask = Image.open(fmask_path)
            bg = Image.new(image.mode, image.size, (0, 0, 0))
            image = Image.composite(image, bg, fmask)

        orig_image = np.array(image)[..., [2, 1, 0]]  # RGB to BGR
        image_path = self.video_path.replace(".mp4", f"/{idx:06d}.webp")
        return image_path, orig_image
