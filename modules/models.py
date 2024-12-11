import torch
import torchvision
from .networks import Resnet, Clip
from .audioVisual_model import AudioVisualModel
import clip


class ModelBuilder():
    # builder for visual stream
    def build_visual(self, pool_type='avgpool', input_channel=3, fc_out=512, weights='', arch_frame='resnet18'):
        pretrained = True
        
        if arch_frame == 'resnet18':
            original_resnet = torchvision.models.resnet18(pretrained)
            net = Resnet(original_resnet, pool_type=pool_type, use_transformer=True)
        else:
            model, _ = clip.load("ViT-B/32", device="cpu")
            net = Clip(model, pool_type=pool_type, use_transformer=True)

        if len(weights) > 0:
            print('Loading weights for visual stream')
            net.load_state_dict(torch.load(weights),strict=True)
        return net

    #builder for audio stream
    def build_unet(self, dim=64, input_nc=1, output_nc=1, weights=''):
        net = AudioVisualModel(dim=dim, input_nc=input_nc, output_nc=output_nc)
        if len(weights) > 0:
            print('Loading weights for UNet')
            net.load_state_dict(torch.load(weights),strict=False)
        return net
