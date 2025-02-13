# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from .misc import NestedTensor

from .position_encoding import build_position_encoding
from COTR.utils import debug_utils, constants


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool, layer='layer3'):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
                #print(f'freeze {name}')
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {layer: "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward_raw(self, x):
        y = self.body(x)
        assert len(y.keys()) == 1
        return y['0']

    def forward(self, tensor_list: NestedTensor):
#         print ('tensor_list_shape' , tensor_list.tensors.shape )
        # assert tensor_list.tensors.shape[-2:] == (256, 1024)
        assert tensor_list.tensors.shape[-2:] == (192, 1280)
        # assert tensor_list.tensors.shape[-2:] == (256, 512)
#         print("tensor_list_left_shape" ,tensor_list.tensors[..., 0:constants.MAX_SIZE].shape)
#         print("tensor_list_right_shape" ,tensor_list.tensors[..., constants.MAX_SIZE: 2*constants.MAX_SIZE].shape)
        left = self.body(tensor_list.tensors[..., 0:512])
        right = self.body(tensor_list.tensors[..., 512: 1024])
        # left = self.body(tensor_list.tensors[..., 0:256])
        # right = self.body(tensor_list.tensors[..., 256: 512])
        xs = {}
        for k in left.keys():
            xs[k] = torch.cat([left[k], right[k]], dim=-1)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool,
                 layer='layer3',
                 num_channels=1024):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=True, norm_layer=FrozenBatchNorm2d)
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers, layer)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        #print("COTR backbone tensor_list type" , type(tensor_list))
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    if hasattr(args, 'lr_backbone'): # checking args object in 'lr_backbone' member
        train_backbone = args.lr_backbone > 0
    else:
        train_backbone = False
    backbone = Backbone(args.backbone, train_backbone, False, args.dilation, layer=args.layer, num_channels=args.dim_feedforward)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
