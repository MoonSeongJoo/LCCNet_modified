

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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

#loading COTR network model
from COTR.COTR_models.cotr_model_moon_Ver11_4 import build
from COTR.utils import utils
# from COTR.inference.sparse_engine_Ver3 import SparseEngine

#loading Monodepth2 network model
import monodepth2.networks
import MonoDEVSNet.networks
from environment import environment as env
from image_processing_unit_Ver11_7 import draw_corrs

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
        # self.load_weights_folder = "/home/seongjoo/work/autocalib/LCCNet_Moon/considering_project/MonoDEVSNet/Pre-trained_network" # kaist gpu server2 251
        # self.load_weights_folder = "/home/pc-3/work/autocalib/considering_project/MonoDEVSNet/Pre-trained_network" # sapeon workstation
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

class regressor(nn.Module) :
    def __init__(self, dropout=0.0 , num_kp=100) :
        super(regressor,self).__init__()

        self.num_kp = num_kp
        self.leakyRELU = nn.LeakyReLU(0.1)
        self.mish = nn.Mish()

        self.fc0_rot_aggr = nn.Linear(self.num_kp * 9 , 1024) # select numer of corresepondence matching point * 2 shape[0] # ========= number of kp (self.num_kp) * 4 ===========
        self.bn0_rot_aggr = nn.BatchNorm1d(1024)
        self.fc0_tarsl_aggr = nn.Linear(self.num_kp * 9 , 1024) # select numer of corresepondence matching point * 2 shape[0] # ========= number of kp (self.num_kp) * 4 ===========
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

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.5)


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
        cycle , _ = self.corr(img_reverse_input, query_reverse)
        cycle[..., 0] = cycle[..., 0] - 0.5
        mask = torch.norm(cycle - query_input, dim=-1) < 10 / 640

        return corrs_pred , cycle , mask, enc_out

# Calibration action prediction part
class CalibActionHead(nn.Module):
    def __init__(self):
        super(CalibActionHead, self).__init__()
        self.activation = nn.ReLU()
        self.input_dim = 123780
        # self.input_dim = 512 # batch * (number of keypoint*corr columns)
        self.head_dim = 128
        
        self.lstm = nn.LSTM(input_size = self.input_dim, hidden_size = 2*self.head_dim, 
                            num_layers = 2, batch_first = True, dropout = 0.5)

        self.emb_r = nn.Sequential(
            nn.Linear(self.head_dim*2, self.head_dim),
            self.activation
        )

        self.emb_t = nn.Sequential(
            nn.Linear(self.head_dim*2, self.head_dim),
            self.activation
        )
        self.action_t = nn.Linear(self.head_dim, 3)
        self.action_r = nn.Linear(self.head_dim, 3)
        
    def forward(self, state, h_n, c_n):
        state = state.view(state.shape[0], -1)

        output, (h_n, c_n) = self.lstm(state.unsqueeze(1), (h_n, c_n))
        emb_t = self.emb_t(output)
        emb_r = self.emb_r(output)
        
        action_mean_t = self.action_t(emb_t).squeeze(1)
        action_mean_r = self.action_r(emb_r).squeeze(1)
        action_mean = [action_mean_t, action_mean_r]
        
        return action_mean, (h_n, c_n)

class GenerateSeq(nn.Module):
    """
        에이전트 네트워크를 기반으로 고정 길이 교정 작업 시퀀스 생성
    """
    def __init__(self, model):
        super(GenerateSeq, self).__init__()
        self.agent = model
    
    def forward(self,ds_pc_source,corr_feature, pos_src, pos_tgt):
        batch_size = ds_pc_source.shape[0]
        trg_seqlen = 3

        # 출력 결과 초기화
        outputs_save_transl=torch.zeros(batch_size,trg_seqlen,3)
        outputs_save_rot=torch.zeros(batch_size,trg_seqlen,3) # agent행동
        exp_outputs_save_transl=torch.zeros(batch_size,trg_seqlen,3)
        exp_outputs_save_rot=torch.zeros(batch_size,trg_seqlen,3) # 전문가들이 조치를 감독합니다.
        h_last = torch.zeros(2, corr_feature.shape[0], 256).cuda()
        c_last = torch.zeros(2, corr_feature.shape[0], 256).cuda() # lstm의 중간 출력
        exp_pos_src = pos_src
        
        # 액션 시퀀스 생성
        for i in range(0, trg_seqlen):
            # 전문가의 움직임
            expert_action = env.expert_step_real(exp_pos_src.detach(), pos_tgt.detach())
            # 에이전트 작업
            actions, hc = self.agent(corr_feature.detach() , h_last.detach(), c_last.detach())
            h_last, c_last = hc[0].detach(), hc[1].detach()
            action_t, action_r = actions[0].unsqueeze(1).detach(), actions[1].unsqueeze(1).detach()
            action_tr = torch.cat([action_t, action_r], dim = 1).detach()
            # 다음 단계 상태
            new_source, pos_src = env.step_continous(ds_pc_source.detach(), action_tr, pos_src.detach()) # new_source는 현재 포인트 클라우드를 기록하는 데만 사용되며 입력 포인트 클라우드(apply_trafo에 의해 결정됨)를 반복적으로 업데이트하지 않습니다.
            _ , exp_pos_src = env.step_continous(ds_pc_source.detach(), expert_action, exp_pos_src)
            # 상태 업데이트
            current_source = new_source
            # depth = lidar_project_depth_batch(current_source, calib, (384, 1280))  # 업데이트된 포인트 클라우드에 해당하는 배치의 깊이 맵
            # depth /= 80
            # get
            exp_outputs_save_transl[:,i,:]=expert_action[:,0]
            exp_outputs_save_rot[:,i,:]=expert_action[:,1]
            outputs_save_transl[:,i,:]=actions[0].squeeze(1)
            outputs_save_rot[:,i,:]=actions[1].squeeze(1)

        return exp_outputs_save_transl, exp_outputs_save_rot, outputs_save_transl, outputs_save_rot, pos_src, current_source

class FeatureFusion(nn.Module):
    def __init__(self):
        super(FeatureFusion, self).__init__()
        self.match_layer = nn.Sequential(
            # Here you can define the structure of match_layer
            # This is just an example and may not fit your actual needs
            nn.Conv2d(312, 160, kernel_size=1),
            nn.BatchNorm2d(160),
            nn.ReLU(),
        )
        
        self.match_block = nn.Sequential(
            # Here you can define the structure of match_block
            # This is just an example and may not fit your actual needs
            nn.Conv2d(160, 160, kernel_size=3, padding=1),
            nn.BatchNorm2d(160),
            nn.ReLU(),
            
            nn.Conv2d(160, 160, kernel_size=1),
        )
        
        self.leakyRELU = nn.LeakyReLU(negative_slope=0.01)
        # Add a dropout layer
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, enc_out):
        
        # corrs_emb = corrs_emb.view(corrs_emb.size()[0], -1 ,corrs_emb.size()[1],corrs_emb.size()[2]) 
         
         # Average pooling over the spatial dimensions to get a tensor of shape [batch_size,num_channels]
        enc_out_avgpool = torch.nn.functional.adaptive_avg_pool2d(enc_out,(enc_out.size()[2],enc_out.size()[3]))
        # enc_out_avgpool = torch.nn.functional.adaptive_avg_pool2d(enc_out,(corrs_emb.size()[2],corrs_emb.size()[3]))
        
        # match_enc = torch.cat((enc_out_avgpool , corrs_emb), dim=1)
        
        match_enc = self.match_layer(enc_out_avgpool)
        
        match_enc_residual = self.match_block(match_enc)

        match_emb = match_enc + match_enc_residual
        match_emb = self.dropout(match_emb)
        match_emb = self.leakyRELU(match_emb)
        
        return match_emb


class DepthCalibTranformer(nn.Module):

    def __init__(self, image_size =64, use_feat_from=1, md=4, use_reflectance=False, dropout=0.0,
                 Action_Func='leakyrelu', attention=False, res_num=50 , num_kp=500 ):
  
        super(DepthCalibTranformer, self).__init__()
        self.num_kp = num_kp

        ##### CORR network #######
        self.corr = COTR(self.num_kp)
        self.feature_emb = FeatureFusion()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # Global Average Pooling
        self.flatten = nn.Flatten()
        
        self.linear_x = nn.Linear(312, 100)
        self.linear_y = nn.Linear(900, 100)

        self.feature_bn = nn.BatchNorm1d(123780)
        self.dropout = nn.Dropout(p=0.5)
   
        self.regressor = CalibActionHead() # LSTM regressor

        self.calib_seq = GenerateSeq(self.regressor)

    def forward(self, ds_pc_source, sbs_img , query_input ,corr_target , pos_src, pos_tgt):
#         print ("------- monodepth2 input information--------" )
        # print ("rgb_input_shape =" , rgb_input.shape)
#         print ("depth_input_shape =" , depth_input.shape)
#         print ("query_input_shape =" , query_input.shape)

#         ####### display input signal #########        
#         plt.figure(figsize=(10, 10))
# #         plt.imshow(cv2.cvtColor(torchvision.utils.make_grid(rgb_input).cpu().numpy() , cv2.COLOR_BGR2RGB))
#         plt.imshow(torchvision.utils.make_grid(sbs_img).permute(1,2,0).cpu().numpy())
#         plt.title("RGB_input", fontsize=22)
#         plt.axis('off')
        
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
        
#         if mask.sum() > 0:
#             ('enter cyclic loss mask sum')
#             cycle_loss = torch.nn.functional.mse_loss(cycle[mask], query[mask])        
        
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
        
        # x = self.avgpool(enc_out)        
        # x = self.flatten(x)
        # x = self.linear_x(x) # now x has shape [batch_size=1 ,feature_emb_dim_100]

        # y = corrs_emb.view(corrs_emb.size(0),-1)   # flatten y to have shape [batch_size = 1,total_features_from_all_embeddings=(num_features_per_embedding*feature_emb_dim)=(9*100)=900]
        # y = self.linear_y(y)   # now y has shape [batch_size=1 ,feature_emb_dim_100]
        # y = self.linear_y(y)
        
        # feature_emb=torch.cat((x,y),dim=-1)   #concatenate along the last dimension [batch,100*9+100]
        # feature_emb=x
        enc_emb = self.feature_emb(enc_out)
        enc_emb = enc_emb.view(enc_emb.size(0), -1)
        corrs_emb = corrs_emb.view(corrs_emb.size(0),-1)
        feature_emb=torch.cat((corrs_emb,enc_emb),dim=-1)
        # feature_emb = self.feature_bn(self.feature_aggr(feature_emb))
        feature_emb =self.dropout(self.feature_bn(feature_emb))
        
        exp_transl_seq, exp_rot_seq, transl_seq, rot_seq, pos_final, current_source = self.calib_seq (ds_pc_source, feature_emb, pos_src, pos_tgt)
        
        return corrs_pred , cycle , mask ,exp_transl_seq, exp_rot_seq, transl_seq, rot_seq, pos_final, current_source 
            

        
        


