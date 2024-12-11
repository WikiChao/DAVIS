import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange

# helpers functions

def create_conv(input_channels, output_channels, kernel, paddings, batch_norm=True, Relu=True, stride=1):
    model = [nn.Conv2d(input_channels, output_channels, kernel, stride = stride, padding = paddings)]
    if(batch_norm):
        model.append(nn.BatchNorm2d(output_channels))

    if(Relu):
        model.append(nn.ReLU())

    return nn.Sequential(*model)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)

class Resnet(nn.Module):
    def __init__(self, original_resnet,pool_type='maxpool', use_transformer=False):
        super(Resnet, self).__init__()
        self.pool_type = pool_type
        self.features = nn.Sequential(
            *list(original_resnet.children())[:-1])
        for param in self.features.parameters():
            param.requires_grad = False

        self.use_transformer = use_transformer
        if use_transformer:
            self.transformer = nn.Transformer(d_model=512, num_encoder_layers=3, num_decoder_layers=1, dim_feedforward=512, batch_first=True)

    def forward(self, x, pool=True):
        x = self.features(x)

        if not pool:
            return x

        if self.pool_type == 'avgpool':
            x = F.adaptive_avg_pool2d(x, 1)
        elif self.pool_type == 'maxpool':
            x = F.adaptive_max_pool2d(x, 1)

        x = x.view(x.size(0), x.size(1))
        return x

    def forward_multiframe(self, x, pool=True):
        (B, C, T, H, W) = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B * T, C, H, W)

        x = self.features(x)

        (_, C, H, W) = x.size()
        x = x.view(B, T, C, H, W)
        x = x[:, 0:1, ...]
        x = x.permute(0, 2, 1, 3, 4)

        x = torch.mean(x, dim=(3,4))

        # transformer
        if self.use_transformer:
            x = self.transformer(x.transpose(1,2), x.transpose(1,2)).transpose(1,2)

        if not pool:
            return x

        x = torch.mean(x, dim=2)

        x = x.view(B, C)
        return x

class Clip(nn.Module):
    def __init__(self, model, pool_type='maxpool', use_transformer=False):
        super(Clip, self).__init__()
        self.pool_type = pool_type
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.use_transformer = use_transformer
        if use_transformer:
            self.temporal_transformer = nn.Transformer(d_model=512, num_encoder_layers=3, num_decoder_layers=1, dim_feedforward=2048, batch_first=True)

            # encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
            # self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # for param in self.temporal_transformer.parameters():
        #     param.requires_grad = False

    def forward(self, x, pool=True):
        x = self.model.encode_image(x)

        return x

    def forward_text(self, x):
        x = self.model.encode_text(x)
        return x
        
    def forward_multiframe(self, x, pool=True):
        (B, C, T, H, W) = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B * T, C, H, W)

        x = self.model.encode_image(x)

        (_, C) = x.size()
        x = x.view(B, T, C).transpose(1,2)

        # transformer
        # x = self.temporal_transformer(x.transpose(1,2)).transpose(1,2)
        x = self.temporal_transformer(x.transpose(1,2), x.transpose(1,2)).transpose(1,2)
        
        if not pool:
            return x

        if self.pool_type == 'avgpool':
            x = torch.mean(x, 2)
        elif self.pool_type == 'maxpool':
            x = torch.max(x, 2)[0]

        return x

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered
