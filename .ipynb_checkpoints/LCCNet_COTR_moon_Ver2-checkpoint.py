
# -------------------------------------------------------------------
# Copyright (C) 2020 Harbin Institute of Technology, China
# Author: Xudong Lv (15B901019@hit.edu.cn)
# Released under Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# http://creativecommons.org/licenses/by-nc-sa/4.0/
# -------------------------------------------------------------------

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
#from models.CMRNet.modules.attention import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import argparse
import os
import os.path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from PIL import Image
import easydict

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

#loading COTR network model
from COTR.COTR_models.cotr_model_moon import build
from COTR.utils import utils, debug_utils
from COTR.inference.sparse_engine_Ver1 import SparseEngine

#loading Monodepth2 network model
import monodepth2.networks
        
cotr_args = easydict.EasyDict({
                "out_dir" : "general_config['out']",
                "load_weights" : "None",
                "load_weights_path" : './COTR/out/default/checkpoint.pth.tar' ,
                "load_weights_freeze" : True ,
                "max_corrs" : 100 ,
                "dim_feedforward" : 1024 , 
                "backbone" : "resnet50" ,
                "hidden_dim" : 256 ,
                "dilation" : False ,
                "dropout" : 0.1 ,
                "nheads" : 8 ,
                "layer" : "layer3" ,
                "enc_layers" : 6 ,
                "dec_layers" : 6 ,
                "position_embedding" : "lin_sine"
                
})

# __all__ = [
#     'calib_net'
# ]

#for cotr model parameter setting
import easydict

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.elu = nn.ELU()
        self.leakyRELU = nn.LeakyReLU(0.1)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leakyRELU(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.leakyRELU(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.leakyRELU(out)

        return out


class SEBottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, reduction=16):
        super(SEBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.leakyRELU = nn.LeakyReLU(0.1)
        # self.attention = SCSElayer(planes * self.expansion, ratio=reduction)
        # self.attention = SElayer(planes * self.expansion, ratio=reduction)
        # self.attention = ECAlayer(planes * self.expansion, ratio=reduction)
        # self.attention = SElayer_conv(planes * self.expansion, ratio=reduction)
        # self.attention = SCSElayer(planes * self.expansion, ratio=reduction)
        # self.attention = ModifiedSCSElayer(planes * self.expansion, ratio=reduction)
        # self.attention = DPCSAlayer(planes * self.expansion, ratio=reduction)
        # self.attention = PAlayer(planes * self.expansion, ratio=reduction)
        # self.attention = CAlayer(planes * self.expansion, ratio=reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leakyRELU(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.leakyRELU(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.attention(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.leakyRELU(out)

        return out


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def myconv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                  groups=1, bias=True),
        nn.LeakyReLU(0.1))


def predict_flow(in_planes):
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=True)


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        # self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.maxpool(self.features[-1]))
        self.features.append(self.encoder.layer1(self.features[-1]))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features


class LCCNet(nn.Module):
    """
    Based on the PWC-DC net. add resnet encoder, dilation convolution and densenet connections
    """

    def __init__(self, image_size =64, use_feat_from=1, md=4, use_reflectance=False, dropout=0.0,
                 Action_Func='leakyrelu', attention=False, res_num=18):
        """
        input: md --- maximum displacement (for correlation. default: 4), after warpping
        """
        super(LCCNet, self).__init__()

        #instantization of Mondepth2 network model
        model_name = "mono_640x192"
        encoder_path = os.path.join("/root/work/LCCNet_Moon/monodepth2/models", model_name, "encoder.pth")
        depth_decoder_path = os.path.join("/root/work/LCCNet_Moon/monodepth2/models", model_name, "depth.pth")
        
        self.encoder = monodepth2.networks.ResnetEncoder(18, False)
        self.depth_decoder = monodepth2.networks.DepthDecoder(num_ch_enc=self.encoder.num_ch_enc, scales=range(4))
        
        loaded_dict_enc = torch.load(encoder_path, map_location='cpu')
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(filtered_dict_enc)

        loaded_dict = torch.load(depth_decoder_path, map_location='cpu')
        self.depth_decoder.load_state_dict(loaded_dict)
               
        self.feed_height = loaded_dict_enc['height']
        self.feed_width = loaded_dict_enc['width']
        
        #print("--------self.feed_height -------" , self.feed_height)
        #print("--------self.feed_width------" , self.feed_width)

        self.encoder.eval()
        self.depth_decoder.eval();
        
        # TODO : change correlation to correspendence Transformer algorithm
        self.corr = build(cotr_args)
        
        # TODO : load COTR pre-trained model    
        if cotr_args.load_weights_path is not None:
            print(f"Loading weights from {cotr_args.load_weights_path}")
            weights = torch.load(cotr_args.load_weights_path, map_location='cpu')['model_state_dict']
            utils.safe_load_weights(self.corr, weights)
        
        if cotr_args.load_weights_freeze is True:
            print("COTR pre-trained weights freeze")
            for param in self.corr.parameters():
                param.requires_grad = False
        
        # cotr muti-zoom estimation for using COTR inference engine
        self.engine = SparseEngine(self.corr, 32, mode='tile')
        
        self.leakyRELU = nn.LeakyReLU(0.1)

        #self.fc1 = nn.Linear(fc_size * 4, 512)
        self.fc1 = nn.Linear(4,256)

        #self.fc1_trasl = nn.Linear(512, 256)
        self.fc1_trasl = nn.Linear(256, 256)
        self.fc1_rot = nn.Linear(256, 256)

        self.fc2_trasl = nn.Linear(256, 3)
        self.fc2_rot = nn.Linear(256, 4)

        self.dropout = nn.Dropout(dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
                    

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, rgb_input, lidar_input):
        
#         print ("------- monodepth2 input information--------" )
#         print ("rgb_outputs_shape =" , rgb_input.shape)
#         print ("lidar_outputs_shape =" , lidar_input.shape)
        
        rgb_show = rgb_input[0].permute(1,2,0).cpu().numpy()
        lidar_show = lidar_input[0].permute(1,2,0).cpu().numpy()
        #H, W = rgb.shape[2:4]
        
#         rgb_temp = rgb_input[0].permute(1,2,0).unsqueeze(0)
#         lidar_temp = lidar_input[0].permute(1,2,0).unsqueeze(0)
        
        #mododepth2 pre-processing
#         rgb_input_torch =[]
#         lidar_input_torch =[]
#         rgb_input = rgb_input.unsqueeze(0)
#         lidar_input = lidar_input.unsqueeze(0)
        
        #display gray to color#
        """
        plt.figure(figsize=(10, 10))
        plt.subplot(211)
        plt.imshow(rgb_show)
        plt.title("RGB_resize", fontsize=22)
        plt.axis('off')

        plt.subplot(212)
        plt.imshow(lidar_show, cmap='magma')
        plt.title("Lidar_resize_color", fontsize=22)
        plt.axis('off');                     
        """
        
        with torch.no_grad():
            rgb_features = self.encoder(rgb_input)
            rgb_outputs = self.depth_decoder(rgb_features)
            lidar_features = self.encoder(lidar_input)
            lidar_outputs = self.depth_decoder(lidar_features)  
        
        rgb_cotr_input = rgb_outputs[("disp", 0)]
        lidar_cotr_input = lidar_outputs[("disp", 0)]
        
#         print ("rgb_outputs_shape =" , rgb_cotr_input.shape)
#         print ("lidar_outputs_shape =" , lidar_cotr_input.shape)
        
        rgb_cotr_input = rgb_cotr_input.squeeze(0)
        lidar_cotr_input = lidar_cotr_input.squeeze(0)
        rgb_cotr_input = rgb_cotr_input.permute(1,2,0)
        lidar_cotr_input = lidar_cotr_input.permute(1,2,0)
        ######## end of mododepth2 batch infernece ############         
        
        
        ##### start of COTR input pre-processing #################
        rgb_cotr_input_np_gray = rgb_cotr_input.cpu().numpy()
        lidar_cotr_input_np_gray = lidar_cotr_input.cpu().numpy()
        
#         print ("------- monodepth2 output information--------" )
#         print ("rgb_outputs_shape =" , rgb_cotr_input_np_gray.shape)
#         print ("lidar_outputs_shape =" , lidar_cotr_input_np_gray.shape)
            
#         for batch_idx in range(64) :
#             rgb_cotr_input_np_color = cv2.cvtColor(rgb_cotr_input_np_gray[batch_idx], cv2.COLOR_GRAY2RGB)
#             lidar_cotr_input_np_color = cv2.cvtColor(lidar_cotr_input_np_gray[batch_idx], cv2.COLOR_GRAY2RGB)
        
#             rgb_cotr_input_np_color_resized = cv2.resize(rgb_cotr_input_np_color, (1024,512), interpolation=cv2.INTER_LINEAR)
#             lidar_cotr_input_np_color_resized = cv2.resize(lidar_cotr_input_np_color, (1024,512), interpolation=cv2.INTER_LINEAR)
# #             print ("------- COTR netowork input information--------" )
# #             print ("rgb_cotr_input_np_color_resized_shape =" , rgb_cotr_input_np_color_resized.shape)
# #             print ("lidar_cotr_input_np_color_resized_shape =" , lidar_cotr_input_np_color_resized.shape)
            
#             ###### display depth image each sensor#############
            
#             plt.figure(figsize=(10, 10))
#             plt.subplot(211)
#             plt.imshow(rgb_cotr_input_np_color_resized , cmap='magma')
#             plt.title("rgb_cotr_input_np_color", fontsize=22)
#             plt.axis('off')

#             plt.subplot(212)
#             plt.imshow(lidar_cotr_input_np_color_resized, cmap='magma')
#             plt.title("lidar_cotr_input_np_color", fontsize=22)
#             plt.axis('off');  
            
#             corrs = self.engine.cotr_corr_multiscale_with_cycle_consistency(rgb_cotr_input_np_color_resized, lidar_cotr_input_np_color_resized, np.linspace(0.5, 0.0625, 4), 1, max_corrs=100, queries_a=None)
    
#             input_rgb_pytorch = transforms.ToTensor()(rgb_cotr_input_np_color_resized)
#             input_lidar_pytorch = transforms.ToTensor()(lidar_cotr_input_np_color_resized)
#             input_rgb_pytorch = input_rgb_pytorch.permute(1,2,0)
#             input_lidar_pytorch = input_lidar_pytorch.permute(1,2,0)
            
#             rgb_input_torch.append(input_rgb_pytorch)
#             lidar_input_torch.append(input_lidar_pytorch)
        
#         rgb_input_cotr = torch.stack(rgb_input_torch)
#         lidar_input_cotr = torch.stack(lidar_input_torch)
#         print ("------- COTR input pre-proccesing information--------" )
#         print ("rgb_cotr_input_np_color_shape =" , rgb_input_cotr.shape)
#         print ("lidar_cotr_input_np_color_shape =" , lidar_input_cotr.shape)
        
        rgb_cotr_input_np_color = cv2.cvtColor(rgb_cotr_input_np_gray, cv2.COLOR_GRAY2RGB)
        lidar_cotr_input_np_color = cv2.cvtColor(lidar_cotr_input_np_gray, cv2.COLOR_GRAY2RGB)   
        rgb_cotr_input_np_color_resized = cv2.resize(rgb_cotr_input_np_color, (1024,512), interpolation=cv2.INTER_LINEAR)
        lidar_cotr_input_np_color_resized = cv2.resize(lidar_cotr_input_np_color, (1024,512), interpolation=cv2.INTER_LINEAR)
#         print ("------- COTR netowork input information--------" )
#         print ("rgb_cotr_input_np_color_resized_shape =" , rgb_cotr_input_np_color_resized.shape)
#         print ("lidar_cotr_input_np_color_resized_shape =" , lidar_cotr_input_np_color_resized.shape)
    
        ###### display depth image each sensor#############
        """
        plt.figure(figsize=(10, 10))
        plt.subplot(211)
        plt.imshow(rgb_cotr_input_np_color_resized , cmap='magma')
        plt.title("rgb_cotr_input_np_color", fontsize=22)
        plt.axis('off')

        plt.subplot(212)
        plt.imshow(lidar_cotr_input_np_color_resized, cmap='magma')
        plt.title("lidar_cotr_input_np_color", fontsize=22)
        plt.axis('off');  
        """      
#         rgb_input_cotr = rgb_input_cotr.cpu().numpy()
#         lidar_input_cotr = lidar_input_cotr.cpu().numpy()
        #Ver2.0 COTR - refined coding 
        corrs = self.engine.cotr_corr_multiscale_with_cycle_consistency(rgb_cotr_input_np_color_resized, lidar_cotr_input_np_color_resized, np.linspace(0.5, 0.0625, 4), 1, max_corrs=100, queries_a=None)
        
        #corrs = torch.from_numpy(corrs)
        corrs = torch.from_numpy(np.asarray(corrs))
        #print("--------corrs_shape-------" , corrs.shape)
        x = self.leakyRELU(corrs)
        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        #print("--------x_shape-------" , x.shape)
        x = x.to('cuda')
        x = x.float()
        x = self.leakyRELU(self.fc1(x))
        #print("--------x1_shape-------" , x.shape)

        transl = self.leakyRELU(self.fc1_trasl(x))
        rot = self.leakyRELU(self.fc1_rot(x))
        transl = self.fc2_trasl(transl)
        rot = self.fc2_rot(rot)
        rot = F.normalize(rot, dim=1)

        return transl, rot
            

        
        


