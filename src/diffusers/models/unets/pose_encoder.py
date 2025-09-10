import torch
import torch.nn as nn
import numpy as np
from torch.nn import init
from einops import rearrange

# Reference to MimicMotion:
# https://github.com/Tencent/MimicMotion/blob/c053153a1d124abae8c08568925ae88debc63001/mimicmotion/modules/pose_net.py


class PoseEncoder(nn.Module):
    def __init__(self, out_channels=320):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
        )

        self.final_proj = nn.Conv2d(in_channels=128, out_channels=out_channels, kernel_size=1)

        self.scale = nn.Parameter(torch.ones(1) * 2.0)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.conv_layers:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                init.normal_(m.weight, mean=0.0, std=np.sqrt(2.0 / n))
                if m.bias is not None:
                    init.zeros_(m.bias)
        init.zeros_(self.final_proj.weight)
        if self.final_proj.bias is not None:
            init.zeros_(self.final_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.final_proj(x)
        return x * self.scale
