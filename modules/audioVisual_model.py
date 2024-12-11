import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from . import networks
from .unet import Unet

class AudioVisualModel(torch.nn.Module):
    def name(self):
        return 'AudioVisualModel'

    def __init__(self, dim=64, input_nc=1, output_nc=1):
        super(AudioVisualModel, self).__init__()

        #initialize model and criterions
        self.dim = dim
        self.out_dim = output_nc
        self.channels = input_nc

        # 2D
        self.net_unet = Unet(dim=self.dim, out_dim=self.out_dim, channels=self.channels, self_condition=True)

        # time mlp
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            networks.SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

    def forward(self, x, t, condition):
        # 2D U-Net
        mix, visual_feature, mix_t = condition

        # predict spectrogram
        spec_prediction = self.net_unet(x, t, x_self_cond=mix, mix_t=mix_t, visual_feat = visual_feature)

        return spec_prediction