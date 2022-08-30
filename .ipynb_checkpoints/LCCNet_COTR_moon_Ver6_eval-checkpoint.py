
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
from torchvision.transforms import functional as tvtf
import math
import argparse
import os
import os.path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from PIL import Image , ImageDraw
import easydict

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

#loading COTR network model
from COTR.COTR_models.cotr_model_moon_Ver5 import build
from COTR.utils import utils, debug_utils
from COTR.inference.sparse_engine_Ver1 import SparseEngine

#loading Monodepth2 network model
import monodepth2.networks
        
cotr_args = easydict.EasyDict({
                "out_dir" : "general_config['out']",
                "load_weights" : "None",
#                 "load_weights_path" : './COTR/out/default/checkpoint.pth.tar' ,
#                "load_weights_path" : '/root/work/COTR/out/model/20220415/76000_0.0008val_checkpoint.pth.tar',
                "load_weights_path" : None ,
                "load_weights_freeze" : False ,
                "max_corrs" : 10000 ,
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
        #encoder_path = os.path.join("/root/work/LCCNet_Moon/monodepth2/models", model_name, "encoder.pth")
        depth_decoder_path = os.path.join("/root/work/LCCNet_Moon/monodepth2/models", model_name, "depth.pth")
        
        self.encoder = monodepth2.networks.ResnetEncoder(18, True , num_input_images = 1)
        self.depth_decoder = monodepth2.networks.DepthDecoder(num_ch_enc=self.encoder.num_ch_enc, scales=range(4))
        
#         loaded_dict_enc = torch.load(encoder_path, map_location='cpu')
#         filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.encoder.state_dict()}
#         self.encoder.load_state_dict(filtered_dict_enc)

        loaded_dict = torch.load(depth_decoder_path, map_location='cpu')
        self.depth_decoder.load_state_dict(loaded_dict)
               
#         self.feed_height = loaded_dict_enc['height']
#         self.feed_width = loaded_dict_enc['width']
        
        #print("--------self.feed_height -------" , self.feed_height)
        #print("--------self.feed_width------" , self.feed_width)
        
        monodepth_load_weights_freeze = True
        
        if monodepth_load_weights_freeze is True:
            print("monodepth pre-trained weights freeze")
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.depth_decoder.parameters():
                param.requires_grad = False
        
#         self.encoder.eval()
#         self.depth_decoder.eval();
        
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
        
        
        self.leakyRELU = nn.LeakyReLU(0.1)

        #self.fc1 = nn.Linear(fc_size * 4, 512)
        self.fc1 = nn.Linear(16000, 256) # select numer of corresepondence matching point * 2 shape[0] # ========= number of kp (self.num_kp) * 4 ===========

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
    
    def two_images_side_by_side(self, img_a, img_b):
        assert img_a.shape == img_b.shape, f'{img_a.shape} vs {img_b.shape}'
        assert img_a.dtype == img_b.dtype
        h, w, c = img_a.shape
#         b,h, w, c = img_a.shape
        canvas = np.zeros((h, 2 * w, c), dtype=img_a.dtype)
#         canvas = np.zeros((b, h, 2 * w, c), dtype=img_a.dtype)
        canvas[:, 0 * w:1 * w, :] = img_a
        canvas[:, 1 * w:2 * w, :] = img_b
#         canvas = np.zeros((b, h, 2 * w, c), dtype=img_a.cpu().numpy().dtype)
#         canvas[:, :, 0 * w:1 * w, :] = img_a.cpu().numpy()
#         canvas[:, :, 1 * w:2 * w, :] = img_b.cpu().numpy()

        #canvas[:, :, : , 0 * w:1 * w] = img_a.cpu().numpy()
        #canvas[:, :, : , 1 * w:2 * w] = img_b.cpu().numpy()
        return canvas

    def draw_corrs(self, imgs, corrs, col=(255, 0, 0)):
        imgs = utils.torch_img_to_np_img(imgs)
        out = []
        for img, corr in zip(imgs, corrs):
            img = np.interp(img, [img.min(), img.max()], [0, 255]).astype(np.uint8)
            img = Image.fromarray(img)
            draw = ImageDraw.Draw(img)
#             corr *= np.array([constants.MAX_SIZE * 2, constants.MAX_SIZE, constants.MAX_SIZE * 2, constants.MAX_SIZE])
            corr *= np.array([1280,384,1280,384])
            for c in corr:
                draw.line(c, fill=col)
            out.append(np.array(img))
        out = np.array(out) / 255.0
        return utils.np_img_to_torch_img(out) , out   
    
    def make_queries(self):
        q_list = []
        MAX_SIZE = 256
        for i in range(MAX_SIZE):
            queries = []
            for j in range(MAX_SIZE * 2):
                queries.append([(j) / (MAX_SIZE * 2), i / MAX_SIZE])
            queries = np.array(queries)
            q_list.append(queries)
            queries = torch.from_numpy(np.concatenate(q_list))[None].float().cuda()
        
        return queries

    def forward(self, rgb_input, query_input ,corr_target):
#         print ("------- monodepth2 input information--------" )
#         print ("rgb_input_shape =" , rgb_input.shape)
#         print ("lidar_input_shape =" , lidar_input.shape)
#         print ("query_input_shape =" , query_input.shape)
               
#         img_input = []
#         img_reverse_input = []
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
        
#         with torch.no_grad():
#             rgb_features = self.encoder(rgb_input)
#             rgb_outputs = self.depth_decoder(rgb_features)
#             lidar_features = self.encoder(lidar_input)
#             lidar_outputs = self.depth_decoder(lidar_features)  
        
#         rgb_features = self.encoder(rgb_input)
#         rgb_outputs = self.depth_decoder(rgb_features)
#         lidar_features = self.encoder(lidar_input)
#         lidar_outputs = self.depth_decoder(lidar_features)   
        
#         rgb_cotr_input = rgb_outputs[("disp", 0)]
#         lidar_cotr_input = lidar_outputs[("disp", 0)]
#         lidar_cotr_input = lidar_input
        
#         print ("rgb_outputs_shape =" , rgb_cotr_input.shape)
#         print ("lidar_outputs_shape =" , lidar_cotr_input.shape)
        
#         rgb_cotr_input = rgb_cotr_input.squeeze(0)
#         lidar_cotr_input = lidar_cotr_input.squeeze(0)
#         rgb_cotr_input = rgb_cotr_input.permute(0,2,3,1)
#         rgb_cotr_input = rgb_input.permute(0,2,3,1)
#         lidar_cotr_input = lidar_cotr_input.permute(0,2,3,1)
        
#         rgb_cotr_input_np_gray = rgb_cotr_input.cpu().numpy()
#         lidar_cotr_input_np_gray = lidar_cotr_input.cpu().numpy()

        
#         for idx in range(len(rgb_cotr_input_np_gray)):
# #             rgb_cotr_input_np_color = cv2.cvtColor(rgb_cotr_input_np_gray[idx], cv2.COLOR_GRAY2RGB)
#             rgb_cotr_input_np_color = rgb_cotr_input_np_gray[idx]
# #             lidar_cotr_input_np_color = cv2.cvtColor(lidar_cotr_input_np_gray[idx], cv2.COLOR_GRAY2RGB)
#             lidar_cotr_input_np_color = lidar_cotr_input_np_gray[idx]
#             sbs_img = self.two_images_side_by_side(rgb_cotr_input_np_color, lidar_cotr_input_np_color)
#             img  = tvtf.normalize(tvtf.to_tensor(sbs_img), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
#             img_reverse = torch.cat([img[..., 640:], img[..., :640]], axis=-1)
            
            ###### display depth image each sensor#############

#             plt.figure(figsize=(10, 10))
#             plt.subplot(311)
#             plt.imshow(rgb_cotr_input_np_color , cmap='magma')
#             plt.title("rgb_cotr_input_np_color", fontsize=22)
#             plt.axis('off')

#             plt.subplot(312)
#             plt.imshow(lidar_cotr_input_np_color, cmap='magma')
#             plt.title("lidar_cotr_input_np_color", fontsize=22)
#             plt.axis('off');  

#             plt.subplot(313)
#             plt.imshow(sbs_img, cmap='magma')
#             plt.title("two_images_side_by_side", fontsize=22)
#             plt.axis('off');  
            
            # batch stack 
#             img_input.append(img)
#             img_reverse_input.append(img_reverse)
        
#         img_input = torch.stack(img_input)
#         img_reverse_input = torch.stack(img_reverse_input)
        
#         img_input = img_input.permute(0,2,3,1)
#         img_reverse_input = img_reverse_input.permute(0,2,3,1)
        
#         print ("img_input_shape =" , img_input.shape)
#         print ("img_reverse_input_shape =" , img_reverse_input.shape)        
        
        ######## end of mododepth2 batch infernece ############ 
        
#         rgb input direct feeding - not depth image #
#         rgb_input = rgb_input.squeeze(0)
#         rgb_input = rgb_input.permute(1,2,0)
#         rgb_input_np = rgb_input.cpu().numpy()
        
        #Ver2.0 COTR - refined coding
#         img           =  img.unsqueeze(dim=0).cuda()
#         img_reverse   =  img_reverse.unsqueeze(dim=0).cuda()
#        query =  self.make_queries()
#         query = query_input
#         print ('###query_shape####' , query.shape)
#         img_input = img_input.cuda()
#         img_cpu = img_input.cpu()
#         img_reverse_input  = img_reverse_input.cuda()
        
#         print ('img_input shape' , img_input.shape)
        img_input =  rgb_input
        query     =  query_input
        corr_target_cpu =  corr_target
        img_cpu   =  img_input.cpu()
        
        corrs = self.corr(img_input, query)['pred_corrs']
#         print ('pred_corrs[0] min ' , torch.min(corrs[:,0]))
#         print ('pred_corrs[0] max ' , torch.max(corrs[:,0]))
#         print ('pred_corrs[1] min ' , torch.min(corrs[:,1]))
#         print ('pred_corrs[1] max ' , torch.max(corrs[:,1]))
        
        ##### display corrs images #############
        corrs_cpu = corrs.cpu().detach().numpy()
        query_cpu = query.cpu().detach().numpy()
        corr_target_cpu = corr_target.cpu().detach().numpy()
        
        pred_corrs = np.concatenate([query_cpu, corrs_cpu], axis=-1)
        pred_corrs , draw_pred_out = self.draw_corrs(img_cpu, pred_corrs)
        
        target_corrs = np.concatenate([query_cpu, corr_target_cpu], axis=-1)
        target_corrs , draw_target_out = self.draw_corrs(img_cpu, target_corrs)
        

#         print ('query_corr shape' , query.shape)
#         print ('query_corr value' , query)            
#         print ('target_corr shape' , corr_target.shape)
#         print ('target_corr value' , corr_target)
#         print ('------------- display start for analysis-------------')
#         plt.figure(figsize=(20, 40))
#         plt.subplot(211)
# #         plt.imshow(torchvision.utils.make_grid(pred_corrs , normalize = True).permute(1,2,0))
#         plt.imshow(np.squeeze(draw_pred_out) , cmap='Greys')
#         plt.title("pred_corrs", fontsize=22)
#         plt.axis('off')
#         plt.show()        

#         plt.figure(figsize=(20, 40))
#         plt.subplot(212)
# #         plt.imshow(torchvision.utils.make_grid(target_corrs , normalize = True).permute(1,2,0))
#         plt.imshow(np.squeeze(draw_target_out) , cmap='Greys' )
#         plt.title("gt_corrs", fontsize=22)
#         plt.axis('off')
#         plt.show()
#         print ('------------- display end for analysis-------------')
        
        img_reverse_input = torch.cat([img_input[..., 640:], img_input[..., :640]], axis=-1)
        ##cyclic loss pre-processing
        query_reverse = corrs.clone()
        query_reverse[..., 0] = query_reverse[..., 0] - 0.5
        cycle = self.corr(img_reverse_input, query_reverse)['pred_corrs']
        cycle[..., 0] = cycle[..., 0] - 0.5
        mask = torch.norm(cycle - query, dim=-1) < 10 / 1280
#         if mask.sum() > 0:
#             ('enter cyclic loss mask sum')
#             cycle_loss = torch.nn.functional.mse_loss(cycle[mask], query[mask])        
        
        #corrs = torch.from_numpy(corrs)
#         print("--------corrs_shape-------" , corrs.shape)
        pred_corrs = torch.cat((query,corrs),dim=-1)
#         print("--------pred_corrs_shape-------" , pred_corrs.shape)
#         x = self.leakyRELU(corrs)
        x = self.leakyRELU(pred_corrs)
        x = x.view(x.shape[0], -1)
#         print("--------x_shape-------" , x.shape)
        x = self.dropout(x)
        x = x.to('cuda')
        x = x.float()
#         print("--------x_shape-------" , x.shape)
        x = self.leakyRELU(self.fc1(x))
        #print("--------x1_shape-------" , x.shape)

        transl = self.leakyRELU(self.fc1_trasl(x))
        rot = self.leakyRELU(self.fc1_rot(x))
        transl = self.fc2_trasl(transl)
        rot = self.fc2_rot(rot)
        rot = F.normalize(rot, dim=1)

        return transl, rot , corrs , cycle , mask
            

        
        


