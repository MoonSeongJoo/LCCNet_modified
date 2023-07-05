
#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -------------------------------------------------------------------
# Copyright (C) 2020 Harbin Institute of Technology, China
# Author: Xudong Lv (15B901019@hit.edu.cn)
# Released under Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# http://creativecommons.org/licenses/by-nc-sa/4.0/
# -------------------------------------------------------------------

import math
import os
import random
import time

# import apex
import mathutils
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn as nn

import torchvision
from torchvision import transforms
from torchvision.transforms import functional as tvtf

from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds

from DatasetLidarCamera_Ver11_0 import DatasetLidarCameraKittiOdometry
from losses_Ver11_0 import DistancePoints3D, GeometricLoss, L1Loss, ProposedLoss, CombinedLoss


from quaternion_distances import quaternion_distance

from tensorboardX import SummaryWriter
from utils import (mat2xyzrpy, merge_inputs, overlay_imgs, quat2mat,
                   quaternion_from_matrix, rotate_back, rotate_forward,
                   tvector2mat)

from image_processing_unit_Ver11_0 import lidar_project_depth , corr_gen , corr_gen_withZ , dense_map , colormap
from LCCNet_COTR_moon_Ver11_0 import LCCNet
#from COTR.inference.sparse_engine_Ver3 import SparseEngine
import warnings
warnings.filterwarnings('ignore')

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False

ex = Experiment("LCCNet" , interactive = True)
ex.captured_out_filter = apply_backspaces_and_linefeeds
from sacred import SETTINGS 
SETTINGS.CONFIG.READ_ONLY_CONFIG = False

np.seterr(invalid='ignore')

# import multiprocessing as mp
# mp.set_start_method('spawn')


# In[2]:


####### dataset path root ############ 
"""
root_dir = "/mnt/data/kitti_odometry"
calib_path ="data_odometry_calib"
image_path ="data_odometry_color"
velodyne_path = "data_odometry_velodyne"
imagegray_path = "data_odometry_gray"
poses_path = "data_odometry_poses"
"""
#######################################
# noinspection PyUnusedLocal
@ex.config
def config():
    checkpoints = './checkpoints/'
    dataset = 'kitti/odom' # 'kitti/raw'
    # data_folder = "/home/ubuntu/data/kitti_odometry"
    # data_folder = "/mnt/sgvrnas/sjmoon/kitti/kitti_odometry"  # kaist gpu server 2 
    # data_folder = "/mnt/sgvrnas/sjmoon/kitti/kitti_odometry"  # kaist gpu server 2 
    # data_folder = "/mnt/data/kitti_odometry" # KAIST GPU server 1
    data_folder = "/mnt/sjmoon/kitti/kitti_odometry"  # sapeon desktop gpu 4090
    use_reflectance = False
    val_sequence = 7
    epochs = 200
    BASE_LEARNING_RATE = 1e-4 # 1e-4
    loss = 'combined'
    max_t = 1.5 # 1.5, 1.0,  0.5,  0.2,  0.1
    max_r = 20.0 # 20.0, 10.0, 5.0,  2.0,  1.0
    batch_size = 15 # 120
    num_worker = 10
    network = 'Res_f1'
    optimizer = 'adamW'
    resume = True
    # weights = '/home/seongjoo/work/autocalib1/considering_project/checkpoints/kitti/odom/val_seq_07/models/checkpoint_r20.00_t1.50_e19_1.885.tar'
    weights = './checkpoints/kitti/odom/val_seq_07/models/checkpoint_r20.00_t1.50_e18_395.210.tar'
    # weights = None
    rescale_rot = 1.0  #LCCNet initail value = 1.0 # value did not use
    rescale_transl = 100.0  #LCCNet initatil value = 2.0 # value did not use
    precision = "O0"
    norm = 'bn'
    dropout = 0.0
    max_depth = 80.
    weight_point_cloud = 0.1 # 이값은 무시해도 됨 loss function에서 직접 관장 원래 LCCNet initail = 0.5
    log_frequency = 1000
    print_frequency = 50
    starting_epoch = 1
    num_kp = 100
    dense_resoltuion = 2
    local_log_frequency = 50 
    ##### re-training option ########
    corr = None # COTR network freeze or not
    regressor_freeze = None
    regressor_init = None

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

EPOCH = 1
def _init_fn(worker_id, seed):
    seed = seed + worker_id + EPOCH*100
    print(f"Init worker {worker_id} with seed {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# In[3]:


# CNN training
@ex.capture
def train(model, optimizer, rgb_input, dense_depth_input, corrs , target_transl, target_rot, loss_fn, point_clouds, loss):
    model.train()

    optimizer.zero_grad()
    
#     ####### display input signal #########        
#     plt.figure(figsize=(10, 10))
# #         plt.imshow(cv2.cvtColor(torchvision.utils.make_grid(rgb_input).cpu().numpy() , cv2.COLOR_BGR2RGB))
#     plt.imshow(torchvision.utils.make_grid(rgb_input).cpu().numpy())
#     plt.title("RGB_input", fontsize=22)
#     plt.axis('off')

    
    queries     = corrs[:, :, :2]
    corr_target = corrs[:, :, 2:]
    
    queries     = queries.cuda().type(torch.float32)
    corr_target = corr_target.cuda().type(torch.float32)

    # Run model
    #transl_err, rot_err = model(rgb_img, refl_img)
    transl_err, rot_err , corr_pred , cycle , mask = model(rgb_input, dense_depth_input, queries , corr_target)
    """
    print("transl_err==========",transl_err)
    print("rot_err=============",rot_err)
    print("point_clouds=============",point_clouds)
    print("target_transl=============",target_transl)
    print("target_rot=============",target_rot)
    """
    
    if loss == 'points_distance' or loss == 'combined':
        losses = loss_fn(point_clouds, target_transl, target_rot, transl_err, rot_err , corr_target , corr_pred , queries ,cycle, mask)
    else:
        losses = loss_fn(target_transl, target_rot, transl_err, rot_err)
    
    #print("losses=============",losses)
    if loss == 'points_distance' or loss == 'combined':
        losses['total_loss'].backward()
    else: 
        losses.backward()
    
    max_norm = 5
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    optimizer.step()

    return losses, rot_err, transl_err


# In[4]:


# CNN test
@ex.capture
def val(model, rgb_input, dense_depth_input ,corrs ,target_transl, target_rot, loss_fn, point_clouds, loss):
    model.eval()

    queries     = corrs[:, :, :2]
    corr_target = corrs[:, :, 2:]

    queries     = queries.cuda().type(torch.float32)
    corr_target = corr_target.cuda().type(torch.float32)
    
    # Run model
    with torch.no_grad():
        #transl_err, rot_err = model(rgb_img, refl_img)
        transl_err, rot_err,corr_pred ,cycle , mask = model(rgb_input,dense_depth_input, queries ,corr_target)

    if loss == 'points_distance' or loss == 'combined':
        losses = loss_fn(point_clouds, target_transl, target_rot, transl_err, rot_err, corr_target , corr_pred ,queries ,cycle, mask)
    else:
        losses = loss_fn(target_transl, target_rot, transl_err, rot_err)

    # if loss != 'points_distance':
    #     total_loss = loss_fn(target_transl, target_rot, transl_err, rot_err)
    # else:
    #     total_loss = loss_fn(point_clouds, target_transl, target_rot, transl_err, rot_err)

    total_trasl_error = torch.tensor(0.0).cuda()
    target_transl = torch.tensor(target_transl).cuda()
    total_rot_error = quaternion_distance(target_rot, rot_err, target_rot.device)
    total_rot_error = total_rot_error * 180. / math.pi
    for j in range(rgb_input.shape[0]):
        total_trasl_error += torch.norm(target_transl[j] - transl_err[j]) * 100.

    # # output image: The overlay image of the input rgb image and the projected lidar pointcloud depth image
    # cam_intrinsic = camera_model[0]
    # rotated_point_cloud =
    # R_predicted = quat2mat(R_predicted[0])
    # T_predicted = tvector2mat(T_predicted[0])
    # RT_predicted = torch.mm(T_predicted, R_predicted)
    # rotated_point_cloud = rotate_forward(rotated_point_cloud, RT_predicted)

    return losses, total_trasl_error.item(), total_rot_error.sum().item(), rot_err, transl_err
#     return losses, total_trasl_error.item(), rot_err, transl_err


# In[5]:


#@ex.automain
@ex.main
def main(_config, _run, seed):
    global EPOCH
    print('Loss Function Choice: {}'.format(_config['loss']))
    print('number of keypoint: {}'.format(_config['num_kp']))
    print('initial learning rate : {}'.format(_config['BASE_LEARNING_RATE']))

    if _config['val_sequence'] is None:
        raise TypeError('val_sequences cannot be None')
    else:
        _config['val_sequence'] = f"{_config['val_sequence']:02d}"
        print("Val Sequence: ", _config['val_sequence'])
        dataset_class = DatasetLidarCameraKittiOdometry
    
    img_shape = (384, 1280) # 네트워크의 입력 규모
    input_size = (256, 512)
    
    _config["checkpoints"] = os.path.join(_config["checkpoints"], _config['dataset'])

    dataset_train = dataset_class(_config['data_folder'], max_r=_config['max_r'], max_t=_config['max_t'],
                                  split='train', use_reflectance=_config['use_reflectance'],
                                  val_sequence=_config['val_sequence'])
    dataset_val = dataset_class(_config['data_folder'], max_r=_config['max_r'], max_t=_config['max_t'],
                                split='val', use_reflectance=_config['use_reflectance'],
                                val_sequence=_config['val_sequence'])
        
    train_dataset_size = len(dataset_train)
    val_dataset_size = len(dataset_val)
    print('Number of the train dataset: {}'.format(train_dataset_size))
    print('Number of the val dataset: {}'.format(val_dataset_size))
    
#     ##### verify - monitoring dataset train ##########
#     rgb = dataset_train[129]['rgb']
#     depth_img =  dataset_train[129]['depth_img']
#     depth_gt_img = dataset_train[129]['depth_gt']


#     plt.figure(figsize=(10, 10))
#     plt.subplot(311)
#     plt.imshow(rgb)
#     plt.title("After RGB_resize", fontsize=22)
#     plt.axis('off')

#     plt.subplot(312)
#     plt.imshow(depth_img, cmap='magma')
#     plt.title("After miscalibrated_Lidar_resize", fontsize=22)
#     plt.axis('off');  
    
#     plt.subplot(313)
#     plt.imshow(depth_gt_img, cmap='magma')
#     plt.title("After Lidar_gt_resize", fontsize=22)
#     plt.axis('off'); 
    
    ####### searching pre-trained model parameter dir ##############
    model_savepath = os.path.join(_config['checkpoints'], 'val_seq_' + _config['val_sequence'], 'models')
    if not os.path.exists(model_savepath):
        os.makedirs(model_savepath)
    log_savepath = os.path.join(_config['checkpoints'], 'val_seq_' + _config['val_sequence'], 'log')
    print ('log_savepath :' , log_savepath)
    if not os.path.exists(log_savepath):
        os.makedirs(log_savepath)
    
    train_writer = SummaryWriter(os.path.join(log_savepath, 'train'))
    val_writer = SummaryWriter(os.path.join(log_savepath, 'val'))

    np.random.seed(seed)
    torch.random.manual_seed(seed)

    def init_fn(x): return _init_fn(x, seed)
    
    # Training and validation set creation
    num_worker = _config['num_worker']
    batch_size = _config['batch_size']
    print("batch_size=" , batch_size)
    TrainImgLoader = torch.utils.data.DataLoader(dataset=dataset_train,
                                                 shuffle=True,
                                                 batch_size=batch_size,
                                                 num_workers=num_worker,
                                                 worker_init_fn=init_fn,
                                                 collate_fn=merge_inputs,
                                                 drop_last=False,
                                                 pin_memory=True)

    ValImgLoader = torch.utils.data.DataLoader(dataset=dataset_val,
                                                shuffle=False,
                                                batch_size=batch_size,
                                                num_workers=num_worker,
                                                worker_init_fn=init_fn,
                                                collate_fn=merge_inputs,
                                                drop_last=False,
                                                pin_memory=True)

    print(len(TrainImgLoader))
    print(len(ValImgLoader))
    
    # loss function choice
    if _config['loss'] == 'simple':
        loss_fn = ProposedLoss(_config['rescale_transl'], _config['rescale_rot'])
    elif _config['loss'] == 'geometric':
        loss_fn = GeometricLoss()
        loss_fn = loss_fn.cuda()
    elif _config['loss'] == 'points_distance':
        loss_fn = DistancePoints3D()
    elif _config['loss'] == 'L1':
        loss_fn = L1Loss(_config['rescale_transl'], _config['rescale_rot'])
    elif _config['loss'] == 'combined':
        loss_fn = CombinedLoss(_config['rescale_transl'], _config['rescale_rot'], _config['weight_point_cloud'])
    else:
        raise ValueError("Unknown Loss Function")
    
    # network choice and settings
    if _config['network'].startswith('Res'):
        feat = 1
        md = 4
        split = _config['network'].split('_')
        for item in split[1:]:
            if item.startswith('f'):
                feat = int(item[-1])
            elif item.startswith('md'):
                md = int(item[2:])
        assert 0 < feat < 7, "Feature Number from PWC have to be between 1 and 6"
        assert 0 < md, "md must be positive"   
        ### netwrok define ####
        model = LCCNet(input_size, use_feat_from=feat, md=md,
                         use_reflectance=_config['use_reflectance'], dropout=_config['dropout'],
                         Action_Func='leakyrelu', attention=False, res_num=50 , num_kp = _config['num_kp'] )

    else:
        raise TypeError("Network unknown")
    if _config['weights'] is not None:
        print(f"Loading weights from {_config['weights']}")
        checkpoint = torch.load(_config['weights'], map_location='cuda:0')
        # checkpoint = torch.load(_config['weights'], map_location='')
        print('-------checkpoint-keys-------',checkpoint.keys() )
        saved_state_dict = checkpoint['state_dict']
        model.load_state_dict(saved_state_dict)

    # COTR network 층을 freeze 시킴
    if _config['corr'] == 'freeze' :    
        for name, param in model.corr.named_parameters():
            param.requires_grad = False
        print (f"COTR network {_config['corr']}")
    elif _config['corr'] == None :  
        print (f"COTR network {_config['corr']}")

    # to-do : regressor last network initailizing only
    if _config['regressor_freeze'] == 'freeze' :
        for name, param in model.named_parameters():
            if name == 'fc1' or name == 'fc1_trasl' or name == 'fc1_rot':
                param.requires_grad = False
        print (f"regreesor network {_config['regressor_freeze']}")
    elif _config['regressor_freeze'] == None :
        print (f"regreesor network {_config['regressor_freeze']}")
    
    if _config['regressor_init'] == 'init' :

        for m in model.fc2_trasl.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
        for m in model.fc2_rot.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
        print (f"regreesor network init {_config['regressor_init']}")
    
    elif _config['regressor_init'] == None :
        print (f"regreesor network init {_config['regressor_init']}")
    
    ########### parallel gpu loding ###########
    model = nn.DataParallel(model)
    model = model.cuda()

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    
    ############ optimizer setting ##############
    if _config['loss'] == 'geometric':
        parameters += list(loss_fn.parameters())
    if _config['optimizer'] == 'adam':
        optimizer = optim.Adam(parameters, lr=_config['BASE_LEARNING_RATE'], weight_decay=5e-6)
        # Probably this scheduler is not used
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200], gamma=0.5)
#         scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 50, 70], gamma=0.3)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 50, 70], gamma=0.5)
    if _config['optimizer'] == 'adamW':
        optimizer = optim.AdamW(parameters, lr=_config['BASE_LEARNING_RATE'], weight_decay=4e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200], gamma=0.5)
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3 , steps_per_epoch=10 ,epochs=_config['epochs'] , anneal_strategy ='cos')
    
    else:
        optimizer = optim.SGD(parameters, lr=_config['BASE_LEARNING_RATE'], momentum=0.9,
                              weight_decay=5e-6, nesterov=True)
    
    ########## training resume setting ###############
    starting_epoch = _config['starting_epoch']
    if _config['weights'] is not None and _config['resume']:
        checkpoint = torch.load(_config['weights'], map_location='cpu')
        opt_state_dict = checkpoint['optimizer']
        optimizer.load_state_dict(opt_state_dict)
        if starting_epoch != 0:
            starting_epoch = checkpoint['epoch']

    ######### training epoch ########################3
    start_full_time = time.time()
    BEST_VAL_LOSS = 10000.
    old_save_filename = None

    train_iter = 0
    val_iter = 0
    
    for epoch in range(starting_epoch, _config['epochs'] + 1):
        EPOCH = epoch
        print('This is %d-th epoch' % epoch)
        # 현재 학습률 확인
        current_lr = optimizer.param_groups[0]['lr']
        print("Current learning rate:", current_lr)
        epoch_start_time = time.time()
        
        total_train_loss = 0.
        sum_corr_loss = 0. 
        sum_point_loss = 0. 
        sum_trans_loss = 0.
        sum_rot_loss = 0.
         
        local_loss = 0.
        local_corr_loss = 0.
        local_rot_loss = 0.
        local_trans_loss = 0.
        local_pcl_loss = 0. 
        
        total_val_loss = 0.
        total_val_t = 0.
        total_val_r = 0.
        rgb_resize_shape = [192,640,3]
        
        if _config['optimizer'] != 'adam' and _config['optimizer'] != 'adamW':
            _run.log_scalar("LR", _config['BASE_LEARNING_RATE'] *
                            math.exp((1 - epoch) * 4e-2), epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = _config['BASE_LEARNING_RATE'] * math.exp((1 - epoch) * 4e-2)
        else:
            #scheduler.step(epoch%100)
            scheduler.step()
            _run.log_scalar("LR", scheduler.get_lr()[0])
    
    ########### traing batch #############################
        time_for_50ep = time.time()
 
        for batch_idx, sample in enumerate(TrainImgLoader):
            print(f'batch {batch_idx+1}/{len(TrainImgLoader)}', end='\r')
            start_time = time.time()
            
            lidar_input = []
            rgb_input = []
            lidar_gt = []
            shape_pad_input = []
            real_shape_input = []
            pc_rotated_input = []
            corrs_input =[]

            # gt pose
            sample['tr_error'] = sample['tr_error'].cuda()
            sample['rot_error'] = sample['rot_error'].cuda()
            
            start_preprocess = time.time()
            for idx in range(len(sample['rgb'])):
                
                real_shape = [sample['rgb'][idx].shape[0], sample['rgb'][idx].shape[1], sample['rgb'][idx].shape[2]]
                rgb = sample['rgb'][idx]
                # rgb = cv2.resize(rgb, (640,192), interpolation=cv2.INTER_LINEAR)
                # rgb = transforms.ToTensor()(rgb)
                rgb = transforms.ToTensor()(rgb).cuda()
                
                #calibrated point cloud 2d projection
                # pc_lidar = sample['point_cloud'][idx].clone()
                pc_lidar = sample['point_cloud'][idx].clone().cuda()
                if _config['max_depth'] < 80.:
                    pc_lidar = pc_lidar[:, pc_lidar[0, :] < _config['max_depth']].clone()
                
                depth_gt, gt_uv , gt_z , gt_points_index  = lidar_project_depth(pc_lidar, sample['calib'][idx], real_shape) # image_shape
                depth_gt /= _config['max_depth']
                
                # mis-calibrated point cloud 2d projection
                R = mathutils.Quaternion(sample['rot_error'][idx]).to_matrix()
                R.resize_4x4()
                T = mathutils.Matrix.Translation(sample['tr_error'][idx])
                RT = T @ R

                pc_rotated = rotate_back(sample['point_cloud'][idx].cuda(), RT) # Pc` = RT * Pc
                if _config['max_depth'] < 80.:
                    pc_rotated = pc_rotated[:, pc_rotated[0, :] < _config['max_depth']].clone()
                
                depth_img, uv , z, points_index = lidar_project_depth(pc_rotated, sample['calib'][idx], real_shape) # image_shape
                depth_img /= _config['max_depth']
                
                # lidarOnImage = np.hstack([uv, z])
                lidarOnImage = torch.cat((uv, z), dim=1)
                dense_depth_img = dense_map(lidarOnImage.T , real_shape[1], real_shape[0] , _config['dense_resoltuion']) # argument = (lidarOnImage.T , 1241, 376 , 8)
                # dense_depth_img = dense_depth_img.astype(np.uint8)
                dense_depth_img = dense_depth_img.to(dtype=torch.uint8)
                dense_depth_img_color = colormap(dense_depth_img)
                # dense_depth_img_color = transforms.ToTensor()(dense_depth_img_color)
                
                # PAD ONLY ON RIGHT AND BOTTOM SIDE
                shape_pad = [0, 0, 0, 0]

                shape_pad[3] = (img_shape[0] - rgb.shape[1])  # // 2
                shape_pad[1] = (img_shape[1] - rgb.shape[2])  # // 2 + 1

                rgb = F.pad(rgb, shape_pad)
                # depth_img = F.pad(depth_img, shape_pad)
                depth_gt = F.pad(depth_gt, shape_pad)
                dense_depth_img_color = F.pad(dense_depth_img_color, shape_pad)
            
                # corr dataset generation 
                # corrs = corr_gen(gt_points_index, points_index , gt_uv, uv , _config["num_kp"])
                corrs_with_z = corr_gen_withZ (gt_points_index, points_index , gt_uv, uv , gt_z, z, real_shape, rgb_resize_shape, _config["num_kp"] )
                # corrs = np.concatenate([corrs_with_z[:,:2],corrs_with_z[:,3:5]], axis=1)
                corrs = torch.cat([corrs_with_z[:, :2], corrs_with_z[:, 3:5]], dim=1)
                # corrs = torch.tensor(corrs)
                
                # batch stack 
                rgb_input.append(rgb)
                lidar_input.append(dense_depth_img_color)
                lidar_gt.append(depth_gt)
                real_shape_input.append(real_shape)
                shape_pad_input.append(shape_pad)
                pc_rotated_input.append(pc_rotated)
                corrs_input.append(corrs)
            
            lidar_input = torch.stack(lidar_input) 
            rgb_input = torch.stack(rgb_input) 
            rgb_show = rgb_input.clone()
            lidar_show = lidar_input.clone()
            rgb_input = F.interpolate(rgb_input, size=[192, 640], mode="bilinear") # lidar 2d depth map input [192,640,1]
            lidar_input = F.interpolate(lidar_input, size=[192, 640], mode="bilinear") # camera input = [192,640,3]
            # lidar_input = torch.cat((lidar_input,lidar_input,lidar_input), 1)
            corrs_input = torch.stack(corrs_input)
            
            
            # ####### display input signal #########        
            # plt.figure(figsize=(10, 10))
            # plt.subplot(211)
            # plt.imshow(torchvision.utils.make_grid(rgb_input).permute(1,2,0).cpu().numpy())
            # plt.title("RGB_input", fontsize=22)
            # plt.axis('off')

            # plt.subplot(212)
            # plt.imshow(torchvision.utils.make_grid(lidar_input).permute(1,2,0).cpu().numpy() , cmap='magma')
            # plt.title("dense_depth_input", fontsize=22)
            # plt.axis('off')        
            # ############# end of display input signal ###################
            
            
            end_preprocess = time.time()

            loss, R_predicted,  T_predicted = train(model, optimizer, rgb_input, lidar_input , corrs_input ,
                                      sample['tr_error'], sample['rot_error'],
                                      loss_fn, sample['point_cloud'], _config['loss'])
        
            if _config['loss'] == 'points_distance' or _config['loss'] == 'combined':
                local_loss += loss['total_loss'].item()
                local_corr_loss += loss['corr_loss'].item()
                local_pcl_loss += loss['point_clouds_loss'].item()
                local_rot_loss += loss['rot_loss'].item()
                local_trans_loss += loss['transl_loss'].item()
                
            else :
                local_loss += loss.item()
            
            train_local_loss = local_loss/50
            train_corr_loss = local_corr_loss/50
            train_pcl_loss = local_pcl_loss/50
            train_rot_loss = local_rot_loss/50
            train_trans_loss =local_trans_loss/50
            

#             if batch_idx % _config['log_frequency'] == 0:
#                 show_idx = 0
#                 # output image: The overlay image of the input rgb image
#                 # and the projected lidar pointcloud depth image
#                 rotated_point_cloud = pc_rotated_input[show_idx]
#                 R_predicted = quat2mat(R_predicted[show_idx])
#                 T_predicted = tvector2mat(T_predicted[show_idx])
#                 RT_predicted = torch.mm(T_predicted, R_predicted)
#                 rotated_point_cloud = rotate_forward(rotated_point_cloud, RT_predicted)

# #                 depth_pred, uv = lidar_project_depth(rotated_point_cloud,
# #                                                     sample['calib'][show_idx],
# #                                                     real_shape_input[show_idx]) # or image_shape
#                 depth_pred, uv = lidar_project_depth(rotated_point_cloud,
#                                                     sample['calib'][show_idx],
#                                                     real_shape) # or image_shape
#                 depth_pred /= _config['max_depth']
# #                 depth_pred = F.pad(depth_pred, shape_pad_input[show_idx])
# #                 print ('depth_pred shape' , depth_pred.shape)
# #                 print ('rgb_show shape' , rgb_show[show_idx].shape)
# #                 print ('lidar_show shape' , lidar_show[show_idx].shape)
# #                 print ('lidar_gt show shape' , lidar_gt_input[show_idx].shape)
#                 depth_pred = depth_pred.permute(1,2,0)
#                 depth_pred_np = depth_pred.cpu().numpy()
#                 depth_pred_np_resized = cv2.resize(depth_pred_np, (640,192), interpolation=cv2.INTER_LINEAR)
#                 depth_pred_np_resized_tensor = transforms.ToTensor()(depth_pred_np_resized)
# #                 depth_pred_np_resized_tensor = depth_pred_np_resized_tensor.permute(1,2,0)
# #                 print ('depth_pred_np_resized_tensor shape' , depth_pred_np_resized_tensor.shape)

# #                 pred_show = overlay_imgs(rgb_show[show_idx], depth_pred_np_resized_tensor.unsqueeze(0))
# #                 input_show = overlay_imgs(rgb_show[show_idx], lidar_show[show_idx].unsqueeze(0))
# #                 gt_show = overlay_imgs(rgb_show[show_idx], lidar_gt_input[show_idx].unsqueeze(0))
                
# #                 print ('pred_show type' , type(pred_show))

# #                 pred_show = torch.from_numpy(np.asarray(pred_show))
# #                 pred_show = pred_show.permute(2, 0, 1)
# #                 input_show = torch.from_numpy(input_show)
# #                 input_show = input_show.permute(2, 0, 1)
# #                 gt_show = torch.from_numpy(gt_show)
# #                 gt_show = gt_show.permute(2, 0, 1)

# #                 train_writer.add_image("rgb_show", rgb_show[show_idx], train_iter)
#                 train_writer.add_image("gt_lidar", lidar_gt_input[show_idx], train_iter)
# #                 train_writer.add_image("miscalibrated_lidar", lidar_show[show_idx], train_iter)
#                 train_writer.add_image("pred_lidar", depth_pred_np_resized_tensor, train_iter)

# #                 if _config['loss'] == 'combined':
# #                     train_writer.add_scalar("Loss_Point_clouds", loss['point_clouds_loss'].item(), train_iter)
# #                     train_writer.add_scalar("correspondence_matching", loss['corr_loss'].item(), train_iter)
# #                     train_writer.add_scalar("Loss_Total", loss['total_loss'].item(), train_iter)
# #                     train_writer.add_scalar("Loss_Translation", loss['transl_loss'].item(), train_iter)
# #                     train_writer.add_scalar("Loss_Rotation", loss['rot_loss'].item(), train_iter)
# #                 else : 
# #                     train_writer.add_scalar("Loss_Total", loss.item(), train_iter)
            
            if batch_idx % 50 == 0 and batch_idx != 0:

                print(f'Iter {batch_idx}/{len(TrainImgLoader)} training loss = {train_local_loss:.6f}, '
                      f'loss of corr = {train_corr_loss:.6f} ,'
                      f'loss of pcl = {train_pcl_loss:.6f} ,'
                      f'loss of rot = {train_rot_loss:.6f} ,'
                      f'loss of trans = {train_trans_loss:.6f} ,'
                      f'time = {(time.time() - start_time)/rgb_input.shape[0]:.4f}, '
                    #   f'time_preprocess = {(end_preprocess-start_preprocess)/rgb_input.shape[0]:.4f}, '
                      f'time for 50 iter: {time.time()-time_for_50ep:.4f} ')
                
                time_for_50ep = time.time()
                _run.log_scalar("Loss", local_loss/50, train_iter)
                local_loss = 0.
                local_corr_loss =  0.
                local_pcl_loss = 0.
                local_rot_loss = 0.
                local_trans_loss = 0.
                
                #train_loss = train_local_loss / len(dataset_train)
                ######### save network model for intermediate verification #####################  
                if train_local_loss < 0.03:
                    #if val_loss < BEST_VAL_LOSS:
                #    BEST_VAL_LOSS = val_loss
                    #_run.result = BEST_VAL_LOSS
                    if _config['rescale_transl'] > 0:
                        _run.result = total_val_t / len(dataset_val)
                    else:
                        _run.result = total_val_t / len(dataset_val)
                        #_run.result = total_val_r / len(dataset_val)
                    savefilename = f'{model_savepath}/checkpoint_r{_config["max_r"]:.2f}_t{_config["max_t"]:.2f}_e{epoch}_{train_local_loss:.3f}.tar'
                    torch.save({
                        'config': _config,
                        'epoch': epoch,
                        # 'state_dict': model.state_dict(), # single gpu
                        'state_dict': model.module.state_dict(), # multi gpu
                        'optimizer': optimizer.state_dict(),
                        'train_loss': total_train_loss / len(dataset_train),
                        'val_loss': total_val_loss / len(dataset_val),
                    }, savefilename)
                    print(f'Model saved as {savefilename}')
                    # if old_save_filename is not None:
                    #     if os.path.exists(old_save_filename):
                    #         os.remove(old_save_filename)
                    # old_save_filename = savefilename                
            
            if _config['loss'] == 'points_distance' or _config['loss'] == 'combined':
                total_train_loss += loss['total_loss'].item() * len(sample['rgb'])
                sum_corr_loss += loss['corr_loss'].item() * len(sample['rgb'])
                sum_point_loss += loss['point_clouds_loss'].item() * len(sample['rgb'])
                sum_trans_loss += loss['transl_loss'].item() * len(sample['rgb'])
                sum_rot_loss += loss['rot_loss'] * len(sample['rgb'])
            else : 
                total_train_loss += loss.item() * len(sample['rgb'])
            train_iter += 1
            # total_iter += len(sample['rgb'])

        print("------------------------------------")
        print('epoch %d total training loss = %.3f' % (epoch, total_train_loss / len(dataset_train)))
        print('epoch %d total corr loss = %.6f' % (epoch, sum_corr_loss / len(dataset_train)))
        print('epoch %d total point loss = %.4f' % (epoch, sum_point_loss / len(dataset_train)))
        print('epoch %d total trans loss = %.4f' % (epoch, sum_trans_loss / len(dataset_train)))
        print('epoch %d total rot loss = %.4f' % (epoch, sum_rot_loss / len(dataset_train)))
        print('Total epoch time = %.2f' % (time.time() - epoch_start_time))
        print("------------------------------------")
        _run.log_scalar("Total training loss", total_train_loss / len(dataset_train), epoch)
        
        if _config['loss'] == 'combined':
            train_writer.add_scalar("Loss_Total", total_train_loss / len(dataset_train), epoch)
            train_writer.add_scalar("correspondence_matching", sum_corr_loss / len(dataset_train) , epoch)
            train_writer.add_scalar("Loss_Point_clouds", sum_point_loss / len(dataset_train), epoch)
            train_writer.add_scalar("Loss_Translation", sum_trans_loss /len(dataset_train), epoch)
            train_writer.add_scalar("Loss_Rotation", sum_rot_loss /len(dataset_train), epoch)
        else : 
            train_writer.add_scalar("Loss_Total", loss.item(), train_iter)

       ## Validation ##
        total_val_loss = 0.
        total_val_t = 0.
        total_val_r = 0.

        local_loss = 0.0

        for batch_idx, sample in enumerate(ValImgLoader):
            print(f'batch {batch_idx+1}/{len(ValImgLoader)}', end='\r')
            start_time = time.time()
            
            lidar_input = []
            rgb_input = []
            lidar_gt = []
            shape_pad_input = []
            real_shape_input = []
            pc_rotated_input = []
            corrs_input =[]
            
            # gt pose
            sample['tr_error'] = sample['tr_error'].cuda()
            sample['rot_error'] = sample['rot_error'].cuda()

            for idx in range(len(sample['rgb'])):
                real_shape = [sample['rgb'][idx].shape[0], sample['rgb'][idx].shape[1], sample['rgb'][idx].shape[2]]
                
                rgb = sample['rgb'][idx]
                rgb = transforms.ToTensor()(rgb).cuda()
                pc_org = sample['point_cloud'][idx].cuda()
                
                #calibrated point cloud 2d projection
                pc_lidar = pc_org.clone().cuda()
                if _config['max_depth'] < 80.:
                    pc_lidar = pc_lidar[:, pc_lidar[0, :] < _config['max_depth']].clone()
                
                depth_gt, gt_uv , gt_z , gt_points_index  = lidar_project_depth(pc_lidar, sample['calib'][idx], real_shape) # image_shape
                depth_gt /= _config['max_depth']
                
                # mis-calibrated point cloud 2d projection
                R = mathutils.Quaternion(sample['rot_error'][idx]).to_matrix()
                R.resize_4x4()
                T = mathutils.Matrix.Translation(sample['tr_error'][idx])
                RT = T @ R

                pc_rotated = rotate_back(sample['point_cloud'][idx].cuda(), RT) # Pc` = RT * Pc
                if _config['max_depth'] < 80.:
                    pc_rotated = pc_rotated[:, pc_rotated[0, :] < _config['max_depth']].clone()
                
                depth_img, uv , z,  points_index = lidar_project_depth(pc_rotated, sample['calib'][idx], real_shape) # image_shape
                depth_img /= _config['max_depth']

                # lidarOnImage = np.hstack([uv, z])
                lidarOnImage = torch.cat((uv, z), dim=1)
                dense_depth_img = dense_map(lidarOnImage.T , real_shape[1], real_shape[0] , _config['dense_resoltuion']) # argument = (lidarOnImage.T , 1241, 376 , 8)
                # dense_depth_img = dense_depth_img.astype(np.uint8)
                dense_depth_img = dense_depth_img.to(dtype=torch.uint8)
                dense_depth_img_color = colormap(dense_depth_img)
                # dense_depth_img_color = transforms.ToTensor()(dense_depth_img_color)                
                
                # PAD ONLY ON RIGHT AND BOTTOM SIDE
                shape_pad = [0, 0, 0, 0]

                shape_pad[3] = (img_shape[0] - rgb.shape[1])  # // 2
                shape_pad[1] = (img_shape[1] - rgb.shape[2])  # // 2 + 1

                rgb = F.pad(rgb, shape_pad)
                # depth_img = F.pad(depth_img, shape_pad).cuda()
                depth_gt = F.pad(depth_gt, shape_pad).cuda()
                dense_depth_img_color = F.pad(dense_depth_img_color, shape_pad).cuda()
                
                # corr dataset generation 
                # corrs = corr_gen(gt_points_index, points_index , gt_uv, uv , _config["num_kp"])
                # corrs = corrs.cuda()
                corrs_with_z = corr_gen_withZ (gt_points_index, points_index , gt_uv, uv , gt_z, z, real_shape, rgb_resize_shape, _config["num_kp"] )
                # corrs = np.concatenate([corrs_with_z[:,:2],corrs_with_z[:,3:5]], axis=1)
                corrs = torch.cat([corrs_with_z[:, :2], corrs_with_z[:, 3:5]], dim=1)
                # corrs = torch.tensor(corrs)
                
                # batch stack 
                rgb_input.append(rgb)
                lidar_input.append(dense_depth_img_color)
                lidar_gt.append(depth_gt)
                real_shape_input.append(real_shape)
                shape_pad_input.append(shape_pad)
                pc_rotated_input.append(pc_rotated)
                corrs_input.append(corrs)           
            
            lidar_input = torch.stack(lidar_input) 
            rgb_input = torch.stack(rgb_input) 
            rgb_input = F.interpolate(rgb_input, size=[192, 640], mode="bilinear") # lidar 2d depth map input [256,512,1]
            lidar_input = F.interpolate(lidar_input, size=[192, 640], mode="bilinear") # camera input = [256,512,3]
            # lidar_input = torch.cat((lidar_input,lidar_input,lidar_input), 1)
            corrs_input = torch.stack(corrs_input)
            
            # ####### display input signal #########        
            # plt.figure(figsize=(10, 10))
            # plt.subplot(211)
            # plt.imshow(torchvision.utils.make_grid(rgb_input).permute(1,2,0).cpu().numpy())
            # plt.title("RGB_input", fontsize=22)
            # plt.axis('off')

            # plt.subplot(212)
            # plt.imshow(torchvision.utils.make_grid(lidar_input).permute(1,2,0).cpu().numpy() , cmap='magma')
            # plt.title("dense_depth_input", fontsize=22)
            # plt.axis('off')        
            # ############# end of display input signal ###################
                        
            loss, trasl_e, rot_e, R_predicted,  T_predicted = val(model, rgb_input, lidar_input, corrs_input ,
                                                                  sample['tr_error'], sample['rot_error'],
                                                                  loss_fn, sample['point_cloud'], _config['loss'])

            total_val_t += trasl_e
            total_val_r += rot_e
            if _config['loss'] == 'points_distance' or _config['loss'] == 'combined':
                local_loss += loss['total_loss'].item()
            else : 
                local_loss += loss.item()

#             if batch_idx % _config['log_frequency'] == 0:
#                 if _config['loss'] == 'combined':
#                     val_writer.add_scalar("Loss_Translation", trasl_e, val_iter)
#                     val_writer.add_scalar("Loss_Rotation", rot_e, val_iter)
#                     val_writer.add_scalar("Total_Loss", loss['total_loss'].item(), val_iter)
#                 else : 
#                     val_writer.add_scalar("Loss_Translation", trasl_e, val_iter)
#                     val_writer.add_scalar("Loss_Rotation", rot_e, val_iter)
#                     val_writer.add_scalar("Total_Loss", loss['total_loss'].item(), val_iter)
            
            if batch_idx % 50 == 0 and batch_idx != 0:
                print('Iter %d val loss = %.3f , time = %.2f' % (batch_idx, local_loss/50.,
                                                                  (time.time() - start_time)/rgb_input.shape[0]))
                local_loss = 0.0
            
            if _config['loss'] == 'points_distance' or _config['loss'] == 'combined':
                total_val_loss += loss['total_loss'].item() * len(sample['rgb'])
            
            else : 
                total_val_loss += loss.item() * len(sample['rgb'])
            
            val_iter += 1

        print("------------------------------------")
        print('total val loss = %.3f' % (total_val_loss / len(dataset_val)))
        print(f'total traslation error: {total_val_t / len(dataset_val)} cm')
        print(f'total rotation error: {total_val_r / len(dataset_val)} °')
        print("------------------------------------")

        _run.log_scalar("Val_Loss", total_val_loss / len(dataset_val), epoch)
        _run.log_scalar("Val_t_error", total_val_t / len(dataset_val), epoch)
        _run.log_scalar("Val_r_error", total_val_r / len(dataset_val), epoch)

        if _config['loss'] == 'combined':
            val_writer.add_scalar("Loss_Translation", total_val_t / len(dataset_val), epoch)
            val_writer.add_scalar("Loss_Rotation", total_val_r / len(dataset_val), epoch)
            val_writer.add_scalar("Total_Loss", total_val_loss / len(dataset_val), epoch)
        else : 
            val_writer.add_scalar("Loss_Translation", trasl_e, val_iter)
            val_writer.add_scalar("Loss_Rotation", rot_e, val_iter)
            val_writer.add_scalar("Total_Loss", loss['total_loss'].item(), val_iter)

   
     # SAVE
        val_loss = total_val_loss / len(dataset_val)
        if epoch % 1 == 0 :
        #if val_loss < BEST_VAL_LOSS:
        #    BEST_VAL_LOSS = val_loss
            #_run.result = BEST_VAL_LOSS
            if _config['rescale_transl'] > 0:
                _run.result = total_val_t / len(dataset_val)
            else:
                _run.result = total_val_t / len(dataset_val)
                #_run.result = total_val_r / len(dataset_val)
            savefilename = f'{model_savepath}/checkpoint_r{_config["max_r"]:.2f}_t{_config["max_t"]:.2f}_e{epoch}_{val_loss:.3f}.tar'
            torch.save({
                'config': _config,
                'epoch': epoch,
                'state_dict': model.state_dict(), # single gpu
                'state_dict': model.module.state_dict(), # multi gpu
                'optimizer': optimizer.state_dict(),
                'train_loss': total_train_loss / len(dataset_train),
                'val_loss': total_val_loss / len(dataset_val),
            }, savefilename)
            print(f'Model saved as {savefilename}')
            if old_save_filename is not None:
                if os.path.exists(old_save_filename):
                    os.remove(old_save_filename)
            old_save_filename = savefilename

    print('full training time = %.2f HR' % ((time.time() - start_full_time) / 3600))
    return _run.result


# In[6]:


ex.run()

