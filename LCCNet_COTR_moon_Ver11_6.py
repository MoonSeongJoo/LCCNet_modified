

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
from COTR.utils import utils, debug_utils
# from COTR.inference.sparse_engine_Ver3 import SparseEngine

#loading Monodepth2 network model
import monodepth2.networks
import MonoDEVSNet.networks.networks
        
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
        self.load_weights_folder = "/home/pc-3/Downloads/AI_network/Pre-trained network/MonoDELSNet/HRNet"
        self.weights_init = "pretrained"

        # checking height and width are multiples of 32
        assert self.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.width % 32 == 0, "'width' must be a multiple of 32"

        self.device = torch.device("cuda")

        self.models = {}

        # Encoder for (depth, segmentation, pose)
        self.models["encoder"] = self.network_selection('encoder')
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
        if model_key == 'encoder':
            # Multiple network architectures
            if 'HRNet' in self.models_fcn_name[model_key]:
                with open(os.path.join('configs', 'hrnet_w' + str(self.opt.num_layers) + '_vk2.yaml'), 'r') as cfg:
                    config = yaml.safe_load(cfg)
                return  MonoDEVSNet.networks.networks.HRNetPyramidEncoder(config).to(self.device)
            elif 'DenseNet' in self.models_fcn_name[model_key]:
                return  MonoDEVSNet.networks.networks.DensenetPyramidEncoder(densnet_version=self.opt.num_layers).to(self.device)
            elif 'ResNet' in self.models_fcn_name[model_key]:
                return  MonoDEVSNet.networks.networks.ResnetEncoder(self.num_layers,
                                              self.weights_init == "pretrained").to(self.device)
            else:
                raise RuntimeError('Choose a depth encoder within available scope')

        elif model_key == 'depth_decoder':
            return  MonoDEVSNet.networks.networks.DepthDecoder(self.models["encoder"].num_ch_enc).to(self.device)

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

class LCCNet_old(nn.Module):
    """
    Based on the PWC-DC net. add resnet encoder, dilation convolution and densenet connections
    """

    def __init__(self, image_size =64, use_feat_from=1, md=4, use_reflectance=False, dropout=0.0,
                 Action_Func='leakyrelu', attention=False, res_num=50 , num_kp=500 ):
        """
        input: md --- maximum displacement (for correlation. default: 4), after warpping
        """
        super(LCCNet_old, self).__init__()
        self.num_kp = num_kp
        
        # self.mono = MonoDepth() # depth estimation by monodepth2
        self.mono = MonoDelsNet() # depth estimation by monodepth2
        # self.STN = STNNet()
        # TODO : change correlation to correspendence Transformer algorithm
        
        self.corr = build(cotr_args)
        
        # # TODO : load COTR pre-trained model    
        # if cotr_args.load_weights_path is not None:
        #     print(f"Loading weights from {cotr_args.load_weights_path}")
        #     weights = torch.load(cotr_args.load_weights_path, map_location='cuda')['model_state_dict']
        #     # weights = torch.load(cotr_args.load_weights_path, map_location='gpu')['model_state_dict']
        #     utils.safe_load_weights(self.corr, weights)
        
        # if cotr_args.load_weights_freeze is True:
        #     print("COTR pre-trained weights freeze")
        #     corr_model =self.corr.eval()
        #     # for param in self.corr.parameters():
        #     #     param.requires_grad = False
        
        # self.corr_engine = SparseEngine(corr_model, 32, mode='tile')
        
        self.leakyRELU = nn.LeakyReLU(0.1)

        #self.fc1 = nn.Linear(fc_size * 4, 512)
        self.fc1 = nn.Linear(self.num_kp * 4 , 256) # select numer of corresepondence matching point * 2 shape[0] # ========= number of kp (self.num_kp) * 4 ===========
        #self.fc1_trasl = nn.Linear(512, 256)
        self.fc1_trasl = nn.Linear(256, 256)
        self.fc1_rot = nn.Linear(256, 256)

        self.fc2_trasl = nn.Linear(256, 3)
        self.fc2_rot = nn.Linear(256, 4)

        self.dropout = nn.Dropout(dropout)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        #         nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
        #         if m.bias is not None:
        #             m.bias.data.zero_()

class LCCNet_old1(nn.Module):
    """
    Based on the PWC-DC net. add resnet encoder, dilation convolution and densenet connections
    """

    def __init__(self, corr_net, image_size =64, use_feat_from=1, md=4, use_reflectance=False, dropout=0.0,
                 Action_Func='leakyrelu', attention=False, res_num=50 , num_kp=500 ):
        """
        input: md --- maximum displacement (for correlation. default: 4), after warpping
        """
        super(LCCNet_old1, self).__init__()
        self.num_kp = num_kp
        
        self.mono = MonoDepth() # depth estimation by monodepth2
        # self.STN = STNNet()
        # TODO : change correlation to correspendence Transformer algorithm
        
        # self.corr = build(cotr_args)
        self.corr = build(cotr_args)
        
        
        self.leakyRELU = nn.LeakyReLU(0.1)

        #self.fc1 = nn.Linear(fc_size * 4, 512)
        self.fc1_rot_aggr = nn.Linear(self.num_kp * 6 , 256) # select numer of corresepondence matching point * 2 shape[0] # ========= number of kp (self.num_kp) * 4 ===========
        self.fc1_tarsl_aggr = nn.Linear(self.num_kp * 6 , 256) # select numer of corresepondence matching point * 2 shape[0] # ========= number of kp (self.num_kp) * 4 ===========
        #self.fc1_trasl = nn.Linear(512, 256)
        self.fc1_trasl = nn.Linear(256, 256)
        self.fc1_rot = nn.Linear(256, 256)

        self.fc2_trasl = nn.Linear(256, 3)
        self.fc2_rot = nn.Linear(256, 4)

        self.dropout = nn.Dropout(dropout)

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

class LCCNet(nn.Module):
    """
    Based on the PWC-DC net. add resnet encoder, dilation convolution and densenet connections
    """

    def __init__(self, image_size =64, use_feat_from=1, md=4, use_reflectance=False, dropout=0.0,
                 Action_Func='leakyrelu', attention=False, res_num=50 , num_kp=500 ):
        """
        input: md --- maximum displacement (for correlation. default: 4), after warpping
        """
        super(LCCNet, self).__init__()
        self.num_kp = num_kp
        
        self.mono = MonoDepth() # depth estimation by monodepth2
        # self.STN = STNNet()
        # TODO : change correlation to correspendence Transformer algorithm
        
        self.corr = build(cotr_args)
        self.regressor = regressor(dropout=0.0, num_kp=self.num_kp)

        # self.corr = corr_net
        
        # self.leakyRELU = nn.LeakyReLU(0.1)
        # self.mish = nn.Mish()

        # #self.fc1 = nn.Linear(fc_size * 4, 512)
        # self.fc0_rot_aggr = nn.Linear(self.num_kp * 9 , 1024) # select numer of corresepondence matching point * 2 shape[0] # ========= number of kp (self.num_kp) * 4 ===========
        # self.bn0_rot_aggr = nn.BatchNorm1d(1024)
        # self.fc0_tarsl_aggr = nn.Linear(self.num_kp * 9 , 1024) # select numer of corresepondence matching point * 2 shape[0] # ========= number of kp (self.num_kp) * 4 ===========
        # #self.fc1_trasl = nn.Linear(512, 256)

        # self.fc0_trasl = nn.Linear(1024, 512)
        # self.fc0_rot = nn.Linear(1024, 512)
        
        # self.fc1_trasl = nn.Linear(512, 256)
        # self.fc1_rot = nn.Linear(512, 256)

        # self.fc2_trasl = nn.Linear(256, 3)
        # self.fc2_rot = nn.Linear(256, 4)

        # self.dropout = nn.Dropout(dropout)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        #         nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
        #         if m.bias is not None:
        #             m.bias.data.zero_()
                    

    # def _make_layer(self, block, planes, blocks, stride=1):
    #     downsample = None
    #     if stride != 1 or self.inplanes != planes * block.expansion:
    #         downsample = nn.Sequential(
    #             conv1x1(self.inplanes, planes * block.expansion, stride),
    #             nn.BatchNorm2d(planes * block.expansion),
    #         )

    #     layers = []
    #     layers.append(block(self.inplanes, planes, stride, downsample))
    #     self.inplanes = planes * block.expansion
    #     for _ in range(1, blocks):
    #         layers.append(block(self.inplanes, planes))

    #     return nn.Sequential(*layers)
    
    def two_images_side_by_side(self, img_a, img_b):
        assert img_a.shape == img_b.shape, f'{img_a.shape} vs {img_b.shape}'
        assert img_a.dtype == img_b.dtype
#         h, w, c = img_a.shape
        b, h, w, c = img_a.shape
#         canvas = np.zeros((h, 2 * w, c), dtype=img_a.dtype)
#         canvas = np.zeros((b, h, 2 * w, c), dtype=img_a.dtype)
#         canvas[:, 0 * w:1 * w, :] = img_a
#         canvas[:, 1 * w:2 * w, :] = img_b
        canvas = np.zeros((b, h, 2 * w, c), dtype=img_a.cpu().numpy().dtype)
        canvas[:, :, 0 * w:1 * w, :] = img_a.cpu().numpy()
        # canvas[:, :, 1 * w:2 * w, :] = img_b.cpu().numpy()
        canvas[:, :, 1 * w:2 * w, :] = img_b.detach().cpu().numpy()

#         canvas[:, :, : , 0 * w:1 * w] = img_a.cpu().numpy()
#         canvas[:, :, : , 1 * w:2 * w] = img_b.cpu().numpy()
        return canvas

    def draw_corrs(self, imgs, corrs, col=(255, 0, 0)):
        imgs = utils.torch_img_to_np_img(imgs)
        out = []
        
        # 삭제하려는 열의 인덱스 리스트
        cols_to_remove = [2, 5] # Python은 0부터 시작하기 때문에 
        # 유지하려는 열들만 선택합니다.
        cols_to_keep = [i for i in range(corrs.shape[2]) if i not in cols_to_remove]
        # index_select 함수를 사용하여 해당 열들만 선택합니다.
        corrs_shrink = corrs[:,:,cols_to_keep]

        for img, corr in zip(imgs, corrs_shrink):
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
    
    def draw_points(self, img, query , mode='640*192'):

        img = Image.fromarray(np.interp(img, [img.min(), img.max()], [0, 255]).astype(np.uint8))
        draw = ImageDraw.Draw(img)

        # 포인트 그리기
        if mode=='640*192':
            query *= np.array([640,192])
        elif mode=='320_*92':
            query *= np.array([320,192])
        
        for (x, y) in query:
            draw.ellipse((x-1, y-1, x+1, y+1), fill='red', outline='red')

        # 정중앙 위치 계산하기
        # 이미지 크기 구하기
        # w, h = img.size
        # query_x = w // 2
        # query_y = h // 2
        
        # radius = 2
        # # 포인트 그리기
        # draw.ellipse((query_x-radius, query_y-radius, query_x+radius, query_y+radius), fill='red', outline='red')
       
        return np.array(img)
    
    def draw_center_point(self, img):
        
        img = Image.fromarray(np.interp(img, [img.min(), img.max()], [0, 255]).astype(np.uint8))
        draw = ImageDraw.Draw(img)

        # 이미지 크기 구하기
        w, h = img.size

        # 정중앙 위치 계산하기
        query_x = w // 2
        query_y = h // 2
        radius = 1
        # 포인트 그리기
        draw.ellipse((query_x-radius, query_y-radius, query_x+radius, query_y+radius), fill='red', outline='red')

        return np.array(img)

    
    # def make_queries(self):
    #     q_list = []
    #     MAX_SIZE = 256
    #     for i in range(MAX_SIZE):
    #         queries = []
    #         for j in range(MAX_SIZE * 2):
    #             queries.append([(j) / (MAX_SIZE * 2), i / MAX_SIZE])
    #         queries = np.array(queries)
    #         q_list.append(queries)
    #         queries = torch.from_numpy(np.concatenate(q_list))[None].float().cuda()
        
    #     return queries
    
    def colormap(self, disp):
        """"Color mapping for disp -- [H, W] -> [3, H, W]"""
        disp_np = disp.cpu().numpy()        # tensor -> numpy
#         disp_np = disp
        vmax = np.percentile(disp_np, 95)
        normalizer = mpl.colors.Normalize(vmin=disp_np.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')  #magma, plasma, etc.
        colormapped_im = (mapper.to_rgba(disp_np)[:, :, :3])
#         colormapped_im = (mapper.to_rgba(disp_np)[:, :, :3]).astype(np.uint8)
        return colormapped_im.transpose(2, 0, 1)
#         return colormapped_im

    def forward(self, rgb_input, depth_input , query_input ,corr_target):
#         print ("------- monodepth2 input information--------" )
        # print ("rgb_input_shape =" , rgb_input.shape)
#         print ("depth_input_shape =" , depth_input.shape)
#         print ("query_input_shape =" , query_input.shape)

#         ####### display input signal #########        
#         plt.figure(figsize=(10, 10))
# #         plt.imshow(cv2.cvtColor(torchvision.utils.make_grid(rgb_input).cpu().numpy() , cv2.COLOR_BGR2RGB))
#         plt.imshow(torchvision.utils.make_grid(rgb_input).permute(1,2,0).cpu().numpy())
#         plt.title("RGB_input", fontsize=22)
#         plt.axis('off')

        rgb_pred_input = []        
        
        with torch.no_grad():
            rgb_features = self.mono.encoder(rgb_input)
            rgb_outputs  = self.mono.depth_decoder(rgb_features)
            
        rgb_depth_pred = rgb_outputs[("disp", 0)]
        
        for idx in range(len(rgb_depth_pred)):
            rgb_pred = rgb_depth_pred[idx].squeeze(0)
            # print ('center of depth value =' , rgb_pred[192//2,640//2] )
            rgb_pred = self.colormap(rgb_pred)
            rgb_pred = torch.from_numpy(rgb_pred)
            # batch stack 
            rgb_pred_input.append(rgb_pred)
        
        rgb_pred_input = torch.stack(rgb_pred_input)
        # print ("rgb_pred_input_shape =" , rgb_pred_input.shape)       
        
        # ####### display input signal #########        
        # plt.figure(figsize=(10, 10))
        # plt.subplot(311)
        # plt.imshow(torchvision.utils.make_grid(rgb_input).permute(1,2,0).cpu().numpy())
        # plt.title("RGB_input", fontsize=22)
        # plt.axis('off')
        # # plt.savefig('RGB_input1.png' ,bbox_inches='tight', pad_inches=0)
        # # plt.show() 
        
        # plt.subplot(312)
        # plt.imshow(torchvision.utils.make_grid(rgb_pred_input).permute(1,2,0).cpu().numpy() , cmap='magma')
        # plt.title("rgb_depth_pred ", fontsize=22)
        # plt.axis('off')
        # # plt.savefig('rgb_depth_pred1.png' ,bbox_inches='tight', pad_inches=0)
        # # plt.show() 
        
        # plt.subplot(313)
        # plt.imshow(torchvision.utils.make_grid(depth_input).permute(1,2,0).cpu().numpy() , cmap='magma')
        # plt.title("dense_depth_input", fontsize=22)
        # plt.axis('off')
        # # plt.show()        
        # ############# end of display input signal ###################
        
        rgb_pred_input = rgb_pred_input.permute(0,2,3,1)
        rgb_pred_input = rgb_pred_input.type(torch.float32)
        # depth_input = depth_input.permute(0,2,3,1)
        depth_input = depth_input.type(torch.float32)
        
        # Spatial Transformer Network forwarding pass
        # 입력을 변환
        # depth_input = self.STN.stn(depth_input)
        depth_input = depth_input.permute(0,2,3,1)
        
        sbs_img = self.two_images_side_by_side(rgb_pred_input, depth_input)
        sbs_img = torch.from_numpy(sbs_img).permute(0,3,1,2)
        sbs_img = tvtf.normalize(sbs_img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


        ########## transformer source image masking #################
        masked_img = sbs_img.clone()
        # 마스킹할 픽셀의 비율을 정의합니다. 여기서는 전체 픽셀 중 약 10%를 마스킹하도록 설정합니다.
        mask_ratio = 0.75
        # 이미지와 동일한 크기의 랜덤한 텐서를 생성합니다.
        random_tensor = torch.rand_like(sbs_img)
        # 생성된 랜덤 텐서에서 mask_ratio보다 작은 값들에 대한 마스크를 만듭니다.
        mask = random_tensor < mask_ratio
        # 원본 이미지에서 해당 마스크 위치의 픽셀 값을 0으로 설정합니다.
        masked_img[mask] = 0

        # img_input =  sbs_img.cuda().type(torch.float32)
        img_input =  masked_img.cuda().type(torch.float32)
        # rgb_pred_input = rgb_pred_input.cuda()
        query_input[:,:,0] = query_input[:,:,0]/2    # recaling points for sbs image resizing
        query_input[:,:,1] = query_input[:,:,1]/2 
        corr_target[:,:,0] = corr_target[:,:,0]/2 + 0.5 # recaling points for sbs image resizing
        corr_target[:,:,1] = corr_target[:,:,1]/2 
        # query_input =  query_input.cuda().type(torch.float32)
        # corr_target =  corr_target.cuda().type(torch.float32)
        
        # print("img_input dtype :" , img_input.dtype)
        # print("query dtype :" , query.dtype)
        # with torch.no_grad():
        corrs_pred = self.corr(img_input, query_input)['pred_corrs']
        
        ############ display each image draw points ###############
        # for batch_idx in range(len(rgb_pred_input)) :
        #     rgb_pred_input_np = rgb_pred_input[batch_idx].cpu().numpy()
        #     depth_input_np = depth_input[batch_idx].cpu().numpy()
        #     query_np = query[batch_idx].cpu().numpy()
        #     corr_target_np = corr_target[batch_idx].cpu().numpy()
        #     # corr_target_np[:, 0] = corr_target_np[:, 0] - 0.5 # normalize for width resizing
        #     draw_points = self.draw_points(rgb_pred_input_np, query_np , mode='640*192')
        #     draw_points1 = self.draw_points(depth_input_np, corr_target_np , mode='640*192')
        #     # draw_points2 = self.draw_center_point(rgb_pred_input_np)
        #     # draw_points2 = self.draw_center_point(depth_input_np)
           
        #     plt.figure(figsize=(10, 20))
        #     plt.imshow(draw_points)
        #     plt.title("draw_query_points", fontsize=22)
        #     plt.axis('off')
        #     plt.show() 
            
        #     plt.figure(figsize=(10, 20))
        #     plt.imshow(draw_points1)
        #     plt.title("draw_target_points", fontsize=22)
        #     plt.axis('off')
        #     plt.show() 
            
            # plt.figure(figsize=(10, 20))
            # plt.imshow(draw_points2)
            # plt.title("draw_center_points", fontsize=22)
            # plt.axis('off')
            # plt.savefig('output1.png' ,bbox_inches='tight', pad_inches=0)
            # plt.show() 
            
            # rgb_pred_input_np1 = cv2.resize(rgb_pred_input_np, (320,192), interpolation=cv2.INTER_LINEAR)
            # depth_input_np1 = cv2.resize(depth_input_np1, (320,192), interpolation=cv2.INTER_LINEAR)
            # query_np[:, 0] = query_np[:, 0]/2 # normalize for width resizing
            # query_np[:, 1] = query_np[:, 1] # normalize for width resizing
            
            # draw_points2 = self.draw_points(rgb_pred_input_np1, query_np, mode='320*192')
            
            # # plt.figure(figsize=(20, 40))
            # plt.imshow(draw_points2)
            # plt.title("draw_query_points_resizing", fontsize=22)
            # plt.axis('off')
            # plt.show() 
            
            # corrs = self.corr_engine.cotr_corr_multiscale_with_cycle_consistency(rgb_pred_input_np1, depth_input_np1, np.linspace(0.5, 0.0625, 4), 1, max_corrs=self.num_kp, queries_a=query_np)
            ############# end of display ################################

#         print ('pred_corrs[0] min ' , torch.min(corrs[:,0]))
#         print ('pred_corrs[0] max ' , torch.max(corrs[:,0]))
#         print ('pred_corrs[1] min ' , torch.min(corrs[:,1]))
#         print ('pred_corrs[1] max ' , torch.max(corrs[:,1]))
        
        # ##### display corrs images #############
        # corr_target_cpu =  corr_target
        # img_cpu   =  img_input.cpu()
        # corrs_cpu = corrs_pred.cpu().detach().numpy()
        # query_cpu = query_input.cpu().detach().numpy()
        # corr_target_cpu = corr_target.cpu().detach().numpy()
        
        # pred_corrs = np.concatenate([query_cpu, corrs_cpu], axis=-1)
        # pred_corrs , draw_pred_out = self.draw_corrs(img_cpu, pred_corrs)
        
        # target_corrs = np.concatenate([query_cpu, corr_target_cpu], axis=-1)
        # target_corrs , draw_target_out = self.draw_corrs(img_cpu, target_corrs)

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
        
        img_reverse_input = torch.cat([img_input[..., 640:], img_input[..., :640]], axis=-1)
        ##cyclic loss pre-processing
        query_reverse = corrs_pred.clone()
        query_reverse[..., 0] = query_reverse[..., 0] - 0.5
        cycle = self.corr(img_reverse_input, query_reverse)['pred_corrs']
        cycle[..., 0] = cycle[..., 0] - 0.5
        mask = torch.norm(cycle - query_input, dim=-1) < 10 / 640
#         if mask.sum() > 0:
#             ('enter cyclic loss mask sum')
#             cycle_loss = torch.nn.functional.mse_loss(cycle[mask], query[mask])        
        
        concat_pred_corrs = torch.cat((query_input,corrs_pred),dim=-1)
        x_diff = concat_pred_corrs[:, :, 0] - concat_pred_corrs[:, :, 3]  # x - x1 차분
        y_diff = concat_pred_corrs[:, :, 1] - concat_pred_corrs[:, :, 4]  # y - y1 차분
        z_diff = concat_pred_corrs[:, :, 2] - concat_pred_corrs[:, :, 5]  # z - z1 차분
        concat_pred_corrs_diff = torch.stack([x_diff, y_diff,z_diff], dim=2)  # 차분 형태로 변환된 A (shape: (batch, 100, 3))

        concatenated_tensors = torch.cat((concat_pred_corrs, concat_pred_corrs_diff), dim=-1)  # shape (batch, 100, 6) 

        # x = self.leakyRELU(concatenated_tensors)
        # x = x.view(x.shape[0], -1)
        # x = self.dropout(x)
        # # x = x.to('cuda')
        # # x = x.float()
        # aggr_rot_x = self.leakyRELU(self.fc1_rot_aggr(x))
        # aggr_transl_x = self.leakyRELU(self.fc1_tarsl_aggr(x))

        # transl = self.leakyRELU(self.fc1_trasl(aggr_transl_x))
        # rot = self.leakyRELU(self.fc1_rot(aggr_rot_x))

        x = self.regressor.mish(concatenated_tensors)
        x = x.view(x.shape[0], -1)
        x = self.regressor.dropout1(x)
        # x = x.to('cuda')
        # x = x.float()
        aggr_rot_x = self.regressor.mish(self.regressor.fc0_rot_aggr(x))
        aggr_rot_x = self.regressor.dropout2(aggr_rot_x)
        aggr_transl_x = self.regressor.mish(self.regressor.fc0_tarsl_aggr(x))
        aggr_transl_x = self.regressor.dropout2(aggr_transl_x)

        transl = self.regressor.mish(self.regressor.fc0_trasl(aggr_transl_x))
        rot = self.regressor.mish(self.regressor.fc0_rot(aggr_rot_x))
        transl = self.regressor.mish(self.regressor.fc1_trasl(transl))
        rot = self.regressor.mish(self.regressor.fc1_rot(rot))
        transl = self.regressor.fc2_trasl(transl)
        rot = self.regressor.fc2_rot(rot)
        rot = F.normalize(rot, dim=1)

        return transl, rot , corrs_pred , cycle , mask
            

        
        


