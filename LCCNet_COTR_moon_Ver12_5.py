

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
import matplotlib as mpl
import matplotlib.cm as cm
from PIL import Image , ImageDraw
import easydict
import cv2
import sys
import yaml
from image_processing_unit_Ver12_5 import (lidar_project_depth , corr_gen , corr_gen_withZ , dense_map , colormap ,two_images_side_by_side ,random_mask ,draw_corrs)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

#loading COTR network model
from COTR.COTR_models.cotr_model_moon_Ver12_0 import build
from COTR.utils import utils, debug_utils
# from COTR.inference.sparse_engine_Ver3 import SparseEngine

#loading Monodepth2 network model
import monodepth2.networks
import MonoDEVSNet.networks

device ='cuda'

cotr_args = easydict.EasyDict({
                "out_dir" : "general_config['out']",
                # "load_weights" : "None",
#                 "load_weights_path" : './COTR/out/default/checkpoint.pth.tar' ,
                # "load_weights_path" : "./models/200_checkpoint.pth.tar",
                "load_weights_path" : None,
                "load_weights_freeze" : False ,
                "max_corrs" : 1000 ,
                "dim_feedforward" : 1024 , 
                "backbone" : "resnet50" ,
                "hidden_dim" : 312 ,
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

class MonoDepth():
    def __init__(self):
        self.model_name         = "mono_resnet50_640x192"
        self.encoder_path       = os.path.join("./monodepth2/models", self.model_name, "encoder.pth")
        self.depth_decoder_path = os.path.join("./monodepth2/models", self.model_name, "depth.pth")
        
        device = torch.device("cuda")
        self.encoder = monodepth2.networks.ResnetEncoder(50, False)
        self.depth_decoder = monodepth2.networks.DepthDecoder(num_ch_enc=self.encoder.num_ch_enc, scales=range(4))
        
        self.loaded_dict_enc = torch.load(self.encoder_path, map_location=device)
        # self.loaded_dict_enc = torch.load(self.encoder_path, map_location='cuda')
        self.filtered_dict_enc = {k: v for k, v in self.loaded_dict_enc.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(self.filtered_dict_enc)
        self.encoder.cuda()
        # self.encoder.to(device)
        # print ('encoder device : ' , next(self.encoder.parameters()).device)

        self.loaded_dict = torch.load(self.depth_decoder_path, map_location=device)
        # self.loaded_dict = torch.load(self.depth_decoder_path, map_location='cuda')
        self.depth_decoder.load_state_dict(self.loaded_dict)
        # self.depth_decoder.to(device)
        self.depth_decoder.cuda()
        # print ('decoder device : ' , next(self.depth_decoder.parameters()).device)
        self.encoder.eval()
        self.depth_decoder.eval()

#### to-do : write modelsnet 
class MonoDelsNet():
    def __init__(self):        
        self.model_name = "HRNet48_sDkitti_depth_rDkitti_depth_trainwithboth_F[0, -1, 1]_leT_dcF_msT_velF_proj_depth_lidar_supervision_v2_hd"
        self.height = 192   
        self.width = 640   
        self.min_depth = 0.1
        self.max_depth = 80.0
        self.num_layers = 48
        self.models_fcn_name =  {
                                    "depth_encoder": "HRNet",
                                    "depth_decoder": "DepthDecoder",
                                    "pose_encoder": "ResnetEncoder",
                                    "pose_decoder": "PoseDecoder",
                                    "domain_classifier": "DomainClassifier",
                                    "disc_dep_cls": "DiscDepCls",
                                    "dis_t": "ImageDiscriminator",
                                    "dis_depth": "DepthDiscriminator",
                                    "dis_s": "ImageDiscriminator",
                                    "gan_s_decoder": "ImageDecoder",
                                    "gan_t_decoder": "ImageDecoder"
                                }
        #self.trainer_name = 'trainer.py'
        # self.load_weights_folder = "/home/seongjoo/work/autocalib/LCCNet_Moon/considering_project/MonoDEVSNet/Pre-trained_network"
        self.load_weights_folder = "./MonoDEVSNet/Pre-trained_network"
        self.weights_init = "pretrained"

        # checking height and width are multiples of 32
        assert self.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.width % 32 == 0, "'width' must be a multiple of 32"

        self.device = torch.device("cuda")

        self.models = {}

        # Encoder for (depth, segmentation, pose)
        self.models["encoder"] = self.network_selection('depth_encoder')
        # Depth decoder
        self.models["depth_decoder"] = self.network_selection('depth_decoder')

        # Loading pretrained model
        if self.load_weights_folder is not None:
            try:
                self.load_pretrained_models()
                print('Loaded MonoDEVSNet trained model')
            except Exception as e:
                print(e)
                print('models not found, start downloading!')
                sys.exit(0)

        # if not os.path.exists(self.opt.log_dir):
        #     os.makedirs(self.opt.log_dir)
    
    # model_key is CaSe SenSiTivE
    def network_selection(self, model_key):
        if model_key == 'depth_encoder':
            # Multiple network architectures
            if 'HRNet' in self.models_fcn_name[model_key]:
                with open(os.path.join('./MonoDEVSNet/configs', 'hrnet_w' + str(self.num_layers) + '_vk2.yaml'), 'r') as cfg:
                    config = yaml.safe_load(cfg)
                return  MonoDEVSNet.networks.HRNetPyramidEncoder(config).to(self.device)
            elif 'DenseNet' in self.models_fcn_name[model_key]:
                return  MonoDEVSNet.networks.DensenetPyramidEncoder(densnet_version=self.opt.num_layers).to(self.device)
            elif 'ResNet' in self.models_fcn_name[model_key]:
                return  MonoDEVSNet.networks.ResnetEncoder(self.num_layers,
                                              self.weights_init == "pretrained").to(self.device)
            else:
                raise RuntimeError('Choose a depth encoder within available scope')

        elif model_key == 'depth_decoder':
            return  MonoDEVSNet.networks.DepthDecoder(self.models["encoder"].num_ch_enc).to(self.device)

        else:
            raise RuntimeError("Don\'t forget to mention what you want!")
    
    def load_pretrained_models(self):
        # Paths to the models
        encoder_path = os.path.join(self.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(self.load_weights_folder, "depth_decoder.pth")

        # Load model weights
        encoder_dict = torch.load(encoder_path, map_location=torch.device('cpu'))
        self.models["encoder"].load_state_dict({k: v for k, v in encoder_dict.items()
                                                if k in self.models["encoder"].state_dict()})
        self.models["depth_decoder"].load_state_dict(torch.load(decoder_path, map_location=torch.device('cpu')))

        # Move network weights from cpu to gpu device
        self.models["encoder"].to(self.device).eval()
        self.models["depth_decoder"].to(self.device).eval()


class STNNet(nn.Module):
    def __init__(self):
        super(STNNet, self).__init__()
        # self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        # self.fc1 = nn.Linear(320, 50)
        # self.fc2 = nn.Linear(50, 10)

        # 공간 변환을 위한 위치 결정 네트워크 (localization-network)
        self.localization = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7 , stride=2 , padding=3 , bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=7 , stride=2 , padding=3 , bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(True)
        )

        # [3 * 2] 크기의 아핀(affine) 행렬에 대해 예측
        self.fc_loc = nn.Sequential(
            nn.Linear(128 * 12 * 40 , 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # 항등 변환(identity transformation)으로 가중치/바이어스 초기화
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # STN의 forward 함수
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 128 * 12 * 40)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

class COTR(nn.Module):
    
    def __init__(self, num_kp=500):
        super(COTR, self).__init__()
        self.num_kp = num_kp
        ##### CORR network #######
        self.corr = build(cotr_args)
    
    def forward(self, sbs_img , query_input):
        
        corrs_pred , enc_out = self.corr(sbs_img, query_input)
        
        img_reverse_input = torch.cat([sbs_img[..., 640:], sbs_img[..., :640]], axis=-1)
        ##cyclic loss pre-processing
        query_reverse = corrs_pred
        query_reverse[..., 0] = query_reverse[..., 0] - 0.5
        cycle,_ = self.corr(img_reverse_input, query_reverse)
        cycle[..., 0] = cycle[..., 0] - 0.5
        mask = torch.norm(cycle - query_input, dim=-1) < 10 / 640

        return corrs_pred , cycle , mask , enc_out
    
class regressor(nn.Module) :
    def __init__(self, dropout=0.0 , num_kp=100) :
        super(regressor,self).__init__()

        self.num_kp = num_kp
        self.leakyRELU = nn.LeakyReLU(0.1)
        self.mish = nn.Mish()
        
        # transformer encoder feature aggregration
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # Global Average Pooling
        self.flatten = nn.Flatten()
        # self.linear = nn.Linear(312, 100) 

        self.fc0_rot_aggr = nn.Linear(self.num_kp*9 + 312 , 1024) # select numer of corresepondence matching point * 2 shape[0] # ========= number of kp (self.num_kp) * 4 ===========
        self.bn0_rot_aggr = nn.BatchNorm1d(1024)
        self.fc0_tarsl_aggr = nn.Linear(self.num_kp*9 + 312  , 1024) # select numer of corresepondence matching point * 2 shape[0] # ========= number of kp (self.num_kp) * 4 ===========
        self.bn0_tarsl_aggr = nn.BatchNorm1d(1024)
 
        self.fc0_trasl = nn.Linear(1024, 512)
        self.bn0_trasl = nn.BatchNorm1d(512)
        self.fc0_rot = nn.Linear(1024, 512)
        self.bn0_rot = nn.BatchNorm1d(512)
        
        self.fc1_trasl = nn.Linear(512, 256)
        self.bn1_trasl = nn.BatchNorm1d(256)
        self.fc1_rot = nn.Linear(512, 256)
        self.bn1_rot = nn.BatchNorm1d(256)

        self.fc2_trasl = nn.Linear(256, 3)
        self.fc2_rot = nn.Linear(256, 4)

        # self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

class DepthCalibTranformer(nn.Module):
    """
    Based on the PWC-DC net. add resnet encoder, dilation convolution and densenet connections
    """

    def __init__(self, image_size =64, use_feat_from=1, md=4, use_reflectance=False, dropout=0.0,
                 Action_Func='leakyrelu', attention=False, res_num=50 , num_kp=500 ):
        """
        input: md --- maximum displacement (for correlation. default: 4), after warpping
        """
        super(DepthCalibTranformer, self).__init__()
        self.num_kp = num_kp
        
        # self.mono = MonoDepth() # depth estimation by monodepth2
        # self.mono = MonoDelsNet() # depth estimation by monodepth2
        # self.STN = STNNet()
   
        self.corr = COTR(self.num_kp)
        # self.corr = build(cotr_args)
        self.regressor = regressor(dropout=0.0, num_kp=self.num_kp)

    def forward(self, rgb_input, depth_input , query_input ,corr_target):

        # rgb_pred_input = []        
        
        # with torch.no_grad():
        #     rgb_features, _ = self.mono.models["encoder"](rgb_input)
        #     rgb_outputs  = self.mono.models["depth_decoder"](rgb_features)
            
        # rgb_depth_pred = rgb_outputs[("disp", 0)]
        
        # for idx in range(len(rgb_depth_pred)):
        #     rgb_pred = rgb_depth_pred[idx].squeeze(0)
        #     rgb_pred = colormap(rgb_pred)
        #     # rgb_pred = torch.from_numpy(rgb_pred)
        #     # batch stack 
        #     rgb_pred_input.append(rgb_pred)
        
        # rgb_pred_input = torch.stack(rgb_pred_input)
        
        # rgb_pred_input = rgb_pred_input.permute(0,2,3,1)
        # rgb_pred_input = rgb_pred_input.type(torch.float32)
        # depth_input = depth_input.type(torch.float32)
        
        # Spatial Transformer Network forwarding pass
        # 입력을 변환
        # depth_input = self.STN.stn(depth_input)
        rgb_input = rgb_input.permute(0,2,3,1)
        depth_input = depth_input.permute(0,2,3,1)
        
        sbs_img = two_images_side_by_side(rgb_input, depth_input)
        sbs_img = torch.from_numpy(sbs_img).permute(0,3,1,2).to(device)
        sbs_img = tvtf.normalize(sbs_img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        # sbs_img_masked = random_mask(sbs_img)

        query_input[:,:,0] = query_input[:,:,0]/2    # recaling points for sbs image resizing
        query_input[:,:,1] = query_input[:,:,1]/2 
        corr_target[:,:,0] = corr_target[:,:,0]/2 + 0.5 # recaling points for sbs image resizing
        corr_target[:,:,1] = corr_target[:,:,1]/2 
        # query_input =  query_input.cuda().type(torch.float32)
        # corr_target =  corr_target.cuda().type(torch.float32)
        
        # print("img_input dtype :" , img_input.dtype)
        # print("query dtype :" , query.dtype)
        # with torch.no_grad():
        corrs_pred ,cycle, mask ,enc_out = self.corr(sbs_img, query_input)
        
        # ##### display corrs images #############
        # corr_target_cpu =  corr_target
        # img_cpu   =  sbs_img.cpu()
        # corrs_cpu = corrs_pred.cpu().detach().numpy()
        # query_cpu = query_input.cpu().detach().numpy()
        # corr_target_cpu = corr_target.cpu().detach().numpy()
        
        # pred_corrs = np.concatenate([query_cpu, corrs_cpu], axis=-1)
        # pred_corrs , draw_pred_out = draw_corrs(img_cpu, pred_corrs)
        
        # target_corrs = np.concatenate([query_cpu, corr_target_cpu], axis=-1)
        # target_corrs , draw_target_out = draw_corrs(img_cpu, target_corrs)

        # print ('------------- display start for analysis-------------')
        # plt.figure(figsize=(20, 40))
        # plt.subplot(211)
        # plt.imshow(torchvision.utils.make_grid(pred_corrs , normalize = True).permute(1,2,0))
        # plt.title("pred_corrs", fontsize=22)
        # plt.axis('off') 
        # plt.show()        

        # plt.figure(figsize=(20, 40))
        # plt.subplot(212)
        # plt.imshow(torchvision.utils.make_grid(target_corrs , normalize = True).permute(1,2,0))
        # plt.title("gt_corrs", fontsize=22)
        # plt.axis('off')
        # plt.show()
        # print ('------------- display end for analysis-------------')
        # ##### end of display corrs images #############
        
        concat_pred_corrs = torch.cat((query_input,corrs_pred),dim=-1)

        concat_pred_corrs[:, :, 0] = concat_pred_corrs[:, :, 0] * 2
        concat_pred_corrs[:, :, 1] = concat_pred_corrs[:, :, 1] * 2
        concat_pred_corrs[:, :, 3] = (concat_pred_corrs[:, :, 3] - 0.5) * 2
        concat_pred_corrs[:, :, 4] = concat_pred_corrs[:, :, 4] * 2

        x_diff = concat_pred_corrs[:, :, 0] - concat_pred_corrs[:, :, 3]  # x - x1 차분
        y_diff = concat_pred_corrs[:, :, 1] - concat_pred_corrs[:, :, 4]  # y - y1 차분
        z_diff = concat_pred_corrs[:, :, 2] - concat_pred_corrs[:, :, 5]  # z - z1 차분
        concat_pred_corrs_diff = torch.stack([x_diff, y_diff,z_diff], dim=2)  # 차분 형태로 변환된 A (shape: (batch, 100, 3))

        corrs_emb = torch.cat((concat_pred_corrs, concat_pred_corrs_diff), dim=-1)  # shape (batch, 100, 6) 

        x = self.regressor.avgpool(enc_out)        
        x = self.regressor.flatten(x)
        # x = self.regressor.linear(x) # now x has shape [batch_size=1 ,feature_emb_dim_100]
        y=corrs_emb.view(corrs_emb.size(0),-1) 

        feature_emb=torch.cat((x,y),dim=-1) 

        # feature_emb = self.regressor.mish(feature_emb)
        # feature_emb = feature_emb.view(feature_emb.shape[0], -1)
        # feature_emb = self.regressor.dropout1(feature_emb)
        # x = x.to('cuda')
        # x = x.float()
        aggr_rot_x = self.regressor.mish(self.regressor.fc0_rot_aggr(feature_emb))
        aggr_rot_x = self.regressor.dropout2(aggr_rot_x)
        aggr_transl_x = self.regressor.mish(self.regressor.fc0_tarsl_aggr(feature_emb))
        aggr_transl_x = self.regressor.dropout2(aggr_transl_x)

        transl = self.regressor.mish(self.regressor.fc0_trasl(aggr_transl_x))
        rot = self.regressor.mish(self.regressor.fc0_rot(aggr_rot_x))
        transl = self.regressor.mish(self.regressor.fc1_trasl(transl))
        rot = self.regressor.mish(self.regressor.fc1_rot(rot))
        transl = self.regressor.fc2_trasl(transl)
        rot = self.regressor.fc2_rot(rot)
        rot = F.normalize(rot, dim=1)

        return transl, rot , corrs_pred , cycle , mask
            

        
        


