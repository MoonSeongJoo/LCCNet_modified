#!/usr/bin/env python
# coding: utf-8

# In[2]:


import csv
import random
import open3d as o3

import cv2
import mathutils
# import matplotlib
# matplotlib.use('Qt5Agg')
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
import torchvision
from torchvision.transforms import functional as tvtf
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from skimage import io
from tqdm import tqdm
import time

#from models.LCCNet import LCCNet

from LCCNet_COTR_moon_Ver12_5 import DepthCalibTranformer , MonoDelsNet

from quaternion_distances import quaternion_distance
from utils import (mat2xyzrpy, merge_inputs, overlay_imgs, quat2mat,
                   quaternion_from_matrix, rotate_back, rotate_forward,
                   tvector2mat)

from torch.utils.data import Dataset
from pykitti import odometry
import pykitti
import pandas as pd
from PIL import Image
from math import radians
from utils import invert_pose
from torchvision import transforms
#sacred read-only function error correct!!
from sacred import SETTINGS 

from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points

SETTINGS.CONFIG.READ_ONLY_CONFIG = False



# In[9]:


class DatasetTest(Dataset):

    def __init__(self, dataset_dir, transform=None, augmentation=False, use_reflectance=False,
                 max_t=1.5, max_r=20., split='random', device='cpu', test_sequence='00', est='rot'):
        super(DatasetTest, self).__init__()
        self.use_reflectance = use_reflectance
        self.maps_folder = ''
        self.device = device
        self.max_r = max_r
        self.max_t = max_t
        self.augmentation = augmentation
        self.root_dir = dataset_dir
        self.transform = transform
        self.split = split
        self.GTs_R = {}
        self.GTs_T = {}
        self.GTs_T_cam02_velo = {}
        self.K = {}
        self.img_shape =(384,1280)
        # self.num_kp = 4000

        self.all_files = []

        self.calib_path ="data_odometry_calib"
        self.image_path ="data_odometry_color"
        self.velodyne_path = "data_odometry_velodyne"
        self.imagegray_path = "data_odometry_gray"
        self.poses_path = "data_odometry_poses"
        self.val_RT_path = "data_odometry_valRT"
        self.test_RT_path = "data_odometry_testRT"
        
        self.calib_path_total = os.path.join(dataset_dir,self.calib_path,"dataset")
        self.image_path_total = os.path.join(dataset_dir,self.image_path,"dataset")
        self.imagegray_path_total = os.path.join(dataset_dir,self.imagegray_path,"dataset")
        self.velodyne_path_total = os.path.join(dataset_dir,self.velodyne_path,"dataset")
        self.poses_path_total = os.path.join(dataset_dir,self.poses_path,"dataset","poses")
        self.val_RT_path_total = os.path.join(dataset_dir,self.val_RT_path,"dataset")
        self.test_RT_path_total = os.path.join(dataset_dir,self.test_RT_path,"dataset")

        seq = test_sequence
        odom = odometry(self.calib_path_total,seq)
        # odom = odometry(self.calib_path_total, self.poses_path_total, seq)
        calib = odom.calib
        T_cam02_velo_np = calib.T_cam2_velo #gt pose from cam02 to velo_lidar (T_cam02_velo: 4x4)
        self.K[seq] = calib.K_cam2 # 3x3
        T_cam02_velo = torch.from_numpy(T_cam02_velo_np)
        GT_R = quaternion_from_matrix(T_cam02_velo[:3, :3])
        GT_T = T_cam02_velo[3:, :3]
        self.GTs_R[seq] = GT_R # GT_R = np.array([row['qw'], row['qx'], row['qy'], row['qz']])
        self.GTs_T[seq] = GT_T # GT_T = np.array([row['x'], row['y'], row['z']])
        self.GTs_T_cam02_velo[seq] = T_cam02_velo_np #gt pose from cam02 to velo_lidar (T_cam02_velo: 4x4)

        image_list = os.listdir(os.path.join(self.image_path_total, 'sequences', seq, 'image_2'))
        image_list.sort()

        for image_name in image_list:
            if not os.path.exists(os.path.join(self.velodyne_path_total, 'sequences', seq, 'velodyne',
                                               str(image_name.split('.')[0])+'.bin')):
                continue
            if not os.path.exists(os.path.join(self.image_path_total, 'sequences', seq, 'image_2',
                                               str(image_name.split('.')[0])+'.png')):
                continue
            self.all_files.append(os.path.join(seq, image_name.split('.')[0]))

        self.test_RT = []
        if split == 'random':
            test_RT_sequences_path = os.path.join(self.test_RT_path_total,"sequences")
            test_RT_file = os.path.join(self.test_RT_path_total, 'sequences',
                                       f'test_RT_seq{test_sequence}_{max_r:.2f}_{max_t:.2f}.csv')
            
            if not os.path.exists(test_RT_sequences_path):
                os.makedirs(test_RT_sequences_path)
            if os.path.exists(test_RT_file):
                print(f'TEST SET: Using this file: {test_RT_file}')
                df_test_RT = pd.read_csv(test_RT_file, sep=',')
                for index, row in df_test_RT.iterrows():
                    self.test_RT.append(list(row))
            else:
                print(f'TEST SET - Not found: {test_RT_file}')
                print("Generating a new one")
                test_RT_file = open(test_RT_file, 'w')
                test_RT_file = csv.writer(test_RT_file, delimiter=',')
                test_RT_file.writerow(['id', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz'])
                for i in range(len(self.all_files)):
                    rotz = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    roty = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    rotx = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    transl_x = np.random.uniform(-max_t, max_t)
                    transl_y = np.random.uniform(-max_t, max_t)
                    # transl_z = np.random.uniform(-max_t, max_t)
                    transl_z = np.random.uniform(-max_t, min(max_t, 1.))
                    test_RT_file.writerow([i, transl_x, transl_y, transl_z,
                                           rotx, roty, rotz])
                    self.test_RT.append([float(i), transl_x, transl_y, transl_z,
                                         rotx, roty, rotz])

            assert len(self.test_RT) == len(self.all_files), "Something wrong with test RTs"
        elif split == 'constant':
            for i in range(len(self.all_files)):
                if est == 'rot':
                    rotz = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    roty = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    rotx = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    transl_x = max_t
                    transl_y = max_t
                    transl_z = max_t
                elif est == 'tran':
                    rotz = max_r * (3.141592 / 180.0)
                    roty = max_r * (3.141592 / 180.0)
                    rotx = max_r * (3.141592 / 180.0)
                    transl_x = np.random.uniform(-max_t, max_t)
                    transl_y = np.random.uniform(-max_t, max_t)
                    transl_z = np.random.uniform(-max_t, max_t)
                # transl_z = np.random.uniform(-max_t, min(max_t, 1.))
                self.test_RT.append([i, transl_x, transl_y, transl_z,
                                     rotx, roty, rotz])

    def get_ground_truth_poses(self, sequence, frame):
        return self.GTs_T[sequence][frame], self.GTs_R[sequence][frame]

    def custom_transform(self, rgb):
        to_tensor = transforms.ToTensor()
        normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

        rgb = to_tensor(rgb)
        rgb = normalization(rgb)
        return rgb

    def get_2D_lidar_projection(self,pcl, cam_intrinsic):
        pcl_xyz = cam_intrinsic @ pcl.T
        pcl_xyz = pcl_xyz.T
        pcl_z = pcl_xyz[:, 2]
        pcl_xyz = pcl_xyz / (pcl_xyz[:, 2, None] + 1e-10)
        pcl_uv = pcl_xyz[:, :2]

        return pcl_uv, pcl_z


#     def lidar_project_depth(self,pc_rotated, cam_calib, img_shape):
#         pc_rotated = pc_rotated[:3, :].detach().cpu().numpy()
# #         cam_intrinsic = cam_calib.numpy()
#         cam_intrinsic = cam_calib
#         pcl_uv, pcl_z = get_2D_lidar_projection(pc_rotated.T, cam_intrinsic)
#         mask = (pcl_uv[:, 0] > 0) & (pcl_uv[:, 0] < img_shape[1]) & (pcl_uv[:, 1] > 0) & (
#                 pcl_uv[:, 1] < img_shape[0]) & (pcl_z > 0)
#         pcl_uv = pcl_uv[mask]
#         pcl_z = pcl_z[mask]
#         pcl_uv = pcl_uv.astype(np.uint32)
#         pcl_z = pcl_z.reshape(-1, 1)
#         depth_img = np.zeros((img_shape[0], img_shape[1], 1))
#         depth_img[pcl_uv[:, 1], pcl_uv[:, 0]] = pcl_z
#         depth_img = torch.from_numpy(depth_img.astype(np.float32))
# #         depth_img = depth_img.cuda()
#         depth_img = depth_img.permute(2, 0, 1)
#         pc_valid = pc_rotated.T[mask]

#         return depth_img, pcl_uv, pc_valid , pcl_z   

    # From Github https://github.com/balcilar/DenseDepthMap
    def dense_map(self, Pts ,n, m, grid):
        ng = 2 * grid + 1

        mX = np.zeros((m,n)) + np.float("inf")
        mY = np.zeros((m,n)) + np.float("inf")
        mD = np.zeros((m,n))
        mX[np.int32(Pts[1]),np.int32(Pts[0])] = Pts[0] - np.round(Pts[0])
        mY[np.int32(Pts[1]),np.int32(Pts[0])] = Pts[1] - np.round(Pts[1])
        mD[np.int32(Pts[1]),np.int32(Pts[0])] = Pts[2]

        KmX = np.zeros((ng, ng, m - ng, n - ng))
        KmY = np.zeros((ng, ng, m - ng, n - ng))
        KmD = np.zeros((ng, ng, m - ng, n - ng))

        for i in range(ng):
            for j in range(ng):
                KmX[i,j] = mX[i : (m - ng + i), j : (n - ng + j)] - grid - 1 +i
                KmY[i,j] = mY[i : (m - ng + i), j : (n - ng + j)] - grid - 1 +i
                KmD[i,j] = mD[i : (m - ng + i), j : (n - ng + j)]
        S = np.zeros_like(KmD[0,0])
        Y = np.zeros_like(KmD[0,0])

        for i in range(ng):
            for j in range(ng):
                s = 1/np.sqrt(KmX[i,j] * KmX[i,j] + KmY[i,j] * KmY[i,j])
                Y = Y + s * KmD[i,j]
                S = S + s

        S[S == 0] = 1
        out = np.zeros((m,n))
        out[grid + 1 : -grid, grid + 1 : -grid] = Y/S
        return out  
    
    def trim_corrs(self, in_corrs):
        length = in_corrs.shape[0]
        if length >= self.num_kp:
            mask = np.random.choice(length, self.num_kp)
            return in_corrs[mask]
        else:
            mask = np.random.choice(length, self.num_kp - length)
            return np.concatenate([in_corrs, in_corrs[mask]], axis=0)
    
    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        item = self.all_files[idx]
        seq = str(item.split('/')[0])
        rgb_name = str(item.split('/')[1])
        img_path = os.path.join(self.image_path_total, 'sequences', seq, 'image_2', rgb_name+'.png')
        lidar_path = os.path.join(self.velodyne_path_total, 'sequences', seq, 'velodyne', rgb_name+'.bin')
        lidar_scan = np.fromfile(lidar_path, dtype=np.float32)
        pc = lidar_scan.reshape((-1, 4))
        valid_indices = pc[:, 0] < -3.
        valid_indices = valid_indices | (pc[:, 0] > 3.)
        valid_indices = valid_indices | (pc[:, 1] < -3.)
        valid_indices = valid_indices | (pc[:, 1] > 3.)
        pc = pc[valid_indices].copy()
        if self.use_reflectance:
            reflectance = pc[:, 3].copy()
            reflectance = torch.from_numpy(reflectance).float()

        RT_torch = self.GTs_T_cam02_velo[seq].astype(np.float32)

        pc_rot = np.matmul(RT_torch, pc.T)
        pc_rot = pc_rot.astype(np.float).T.copy()
        pc_in = torch.from_numpy(pc_rot.astype(np.float32))#.float()

        if pc_in.shape[1] == 4 or pc_in.shape[1] == 3:
            pc_in = pc_in.t()
        if pc_in.shape[0] == 3:
            homogeneous = torch.ones(pc_in.shape[1]).unsqueeze(0)
            pc_in = torch.cat((pc_in, homogeneous), 0)
        elif pc_in.shape[0] == 4:
            if not torch.all(pc_in[3,:] == 1.):
                pc_in[3,:] = 1.
        else:
            raise TypeError("Wrong PointCloud shape")

        # img = Image.open(img_path)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # try:
        #     img = self.custom_transform(img)
        # except OSError:
        #     new_idx = np.random.randint(0, self.__len__())
        #     return self.__getitem__(new_idx)


        initial_RT = self.test_RT[idx]
        rotz = initial_RT[6]
        roty = initial_RT[5]
        rotx = initial_RT[4]
        transl_x = initial_RT[1]
        transl_y = initial_RT[2]
        transl_z = initial_RT[3]

        #R = mathutils.Euler((rotx, roty, rotz), 'XYZ')
        R = mathutils.Euler((rotx, roty, rotz))
        T = mathutils.Vector((transl_x, transl_y, transl_z))

        R, T = invert_pose(R, T)
        R_torch, T_torch = torch.tensor(R), torch.tensor(T)
        calib = self.K[seq]  
        
        sample = {'rgb': img, 'point_cloud': pc_in, 'calib': calib,
                  'tr_error': T_torch, 'rot_error': R_torch, 'seq': int(seq), 'img_path': img_path,
                  'rgb_name': rgb_name + '.png', 'item': item, 'extrin': RT_torch,
                  'initial_RT': initial_RT}

        return sample

class DatasetLidarCameraKittiRaw(Dataset):

    def __init__(self, dataset_dir, transform=None, augmentation=False, use_reflectance=False,
                 max_t=1.5, max_r=15.0, split='val', device='cpu', val_sequence='2011_09_29_drive_0004_sync'):
        super(DatasetLidarCameraKittiRaw, self).__init__()
        self.use_reflectance = use_reflectance
        self.maps_folder = ''
        self.device = device
        self.max_r = max_r
        self.max_t = max_t
        self.augmentation = augmentation
        self.root_dir = dataset_dir #/data/kitti/raw_data/
        self.transform = transform
        self.split = split
        self.GTs_R = {}
        self.GTs_T = {}
        self.GTs_T_cam02_velo = {}
        self.max_depth = 80
        self.K_list = {}

        self.all_files = []
        date_list = ['2011_09_26', '2011_09_28', '2011_09_29', '2011_09_30', '2011_10_03']
        data_drive_list = ['0001', '0002', '0004', '0016', '0027']
        self.calib_date = {}
        print('val_sequence:', val_sequence)

        for i in range(len(date_list)):
            date = date_list[i]
            data_drive = data_drive_list[i]
            data = pykitti.raw(self.root_dir, date, data_drive) # /data/kitti/raw_data/2011_09_29/2011_09_29_drive_0004_sync
            calib = {'K2': data.calib.K_cam2, 'K3': data.calib.K_cam3,
                     'RT2': data.calib.T_cam2_velo, 'RT3': data.calib.T_cam3_velo}
            self.calib_date[date] = calib

        # date = val_sequence[:10]
        # seq = val_sequence
        # image_list = os.listdir(os.path.join(dataset_dir, date, seq, 'image_02/data'))
        # image_list.sort()
        
        # for image_name in image_list:
        #     if not os.path.exists(os.path.join(dataset_dir, date, seq, 'velodyne_points/data',
        #                                        str(image_name.split('.')[0]) + '.bin')):
        #         continue
        #     if not os.path.exists(os.path.join(dataset_dir, date, seq, 'image_02/data',
        #                                        str(image_name.split('.')[0]) + '.jpg')):  # png
        #         continue
        #     print ("enter")
        #     self.all_files.append(os.path.join(date, seq, 'image_02/data', image_name.split('.')[0]))


        date = val_sequence[:10]
        test_list = ['2011_09_26_drive_0005_sync', '2011_09_26_drive_0070_sync', '2011_09_29_drive_0004_sync' ,'2011_09_30_drive_0016_sync','2011_10_03_drive_0027_sync']
        seq_list = os.listdir(os.path.join(self.root_dir, date))

        for seq in seq_list:
            if not os.path.isdir(os.path.join(dataset_dir, date, seq)):
                continue
            image_list = os.listdir(os.path.join(dataset_dir, date, seq, 'image_02/data'))
            image_list.sort()

            for image_name in image_list:
                if not os.path.exists(os.path.join(dataset_dir, date, seq, 'velodyne_points/data',
                                                   str(image_name.split('.')[0])+'.bin')):
                    continue
                # if not os.path.exists(os.path.join(dataset_dir, date, seq, 'image_02/data',
                #                                    str(image_name.split('.')[0])+'.jpg')): # png
                #     continue
                if seq == val_sequence and (not split == 'train'):
                    self.all_files.append(os.path.join(date, seq, 'image_02/data', image_name.split('.')[0]))
                elif (not seq == val_sequence) and split == 'train' and seq not in test_list:
                    self.all_files.append(os.path.join(date, seq, 'image_02/data', image_name.split('.')[0]))

        self.val_RT = []
        if split == 'val' or split == 'test':
            val_RT_file = os.path.join(dataset_dir,
                                       f'val_RT_seq{val_sequence}_{max_r:.2f}_{max_t:.2f}.csv')
            if os.path.exists(val_RT_file):
                print(f'VAL SET: Using this file: {val_RT_file}')
                df_test_RT = pd.read_csv(val_RT_file, sep=',')
                for index, row in df_test_RT.iterrows():
                    self.val_RT.append(list(row))
            else:
                print(f'TEST SET - Not found: {val_RT_file}')
                print("Generating a new one")
                val_RT_file = open(val_RT_file, 'w')
                val_RT_file = csv.writer(val_RT_file, delimiter=',')
                val_RT_file.writerow(['id', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz'])
                for i in range(len(self.all_files)):
                    rotz = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    roty = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    rotx = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    transl_x = np.random.uniform(-max_t, max_t)
                    transl_y = np.random.uniform(-max_t, max_t)
                    transl_z = np.random.uniform(-max_t, max_t)
                    # transl_z = np.random.uniform(-max_t, min(max_t, 1.))
                    val_RT_file.writerow([i, transl_x, transl_y, transl_z,
                                           rotx, roty, rotz])
                    self.val_RT.append([float(i), transl_x, transl_y, transl_z,
                                         rotx, roty, rotz])

            assert len(self.val_RT) == len(self.all_files), "Something wrong with test RTs"

    def get_ground_truth_poses(self, sequence, frame):
        return self.GTs_T[sequence][frame], self.GTs_R[sequence][frame]

    def __len__(self):
        return len(self.all_files)

    # self.all_files.append(os.path.join(date, seq, 'image_2/data', image_name.split('.')[0]))
    def __getitem__(self, idx):
        item = self.all_files[idx]
        date = str(item.split('/')[0])
        seq = str(item.split('/')[1])
        rgb_name = str(item.split('/')[4])
        # img_path = os.path.join(self.root_dir, date, seq, 'image_02/data', rgb_name+'.jpg') # jpg
        img_path = os.path.join(self.root_dir, date, seq, 'image_02/data', rgb_name+'.png') # png
        lidar_path = os.path.join(self.root_dir, date, seq, 'velodyne_points/data', rgb_name+'.bin')
        lidar_scan = np.fromfile(lidar_path, dtype=np.float32)
        pc = lidar_scan.reshape((-1, 4))
        valid_indices = pc[:, 0] < -3.
        valid_indices = valid_indices | (pc[:, 0] > 3.)
        valid_indices = valid_indices | (pc[:, 1] < -3.)
        valid_indices = valid_indices | (pc[:, 1] > 3.)
        pc = pc[valid_indices].copy()
        pc_lidar = pc.copy()
        pc_org = torch.from_numpy(pc.astype(np.float32))
        # if self.use_reflectance:
        #     reflectance = pc[:, 3].copy()
        #     reflectance = torch.from_numpy(reflectance).float()

        calib = self.calib_date[date]
        RT_cam02 = calib['RT2'].astype(np.float32)
        # camera intrinsic parameter
        calib_cam02 = calib['K2']  # 3x3

        E_RT = RT_cam02

        if pc_org.shape[1] == 4 or pc_org.shape[1] == 3:
            pc_org = pc_org.t()
        if pc_org.shape[0] == 3:
            homogeneous = torch.ones(pc_org.shape[1]).unsqueeze(0)
            pc_org = torch.cat((pc_org, homogeneous), 0)
        elif pc_org.shape[0] == 4:
            if not torch.all(pc_org[3, :] == 1.):
                pc_org[3, :] = 1.
        else:
            raise TypeError("Wrong PointCloud shape")

        pc_rot = np.matmul(E_RT, pc_org.numpy())
        pc_rot = pc_rot.astype(np.float32).copy()
        pc_in = torch.from_numpy(pc_rot)

        h_mirror = False
        # if np.random.rand() > 0.5 and self.split == 'train':
        #     h_mirror = True
        #     pc_in[0, :] *= -1

        # img = Image.open(img_path)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # print(img_path)
        # img = cv2.imread(img_path)
        img_rotation = 0.
        # if self.split == 'train':
        #     img_rotation = np.random.uniform(-5, 5)
        # try:
        #     img = self.custom_transform(img, img_rotation, h_mirror)
        # except OSError:
        #     new_idx = np.random.randint(0, self.__len__())
        #     return self.__getitem__(new_idx)

        # Rotate PointCloud for img_rotation
        # if self.split == 'train':
        #     R = mathutils.Euler((radians(img_rotation), 0, 0), 'XYZ')
        #     T = mathutils.Vector((0., 0., 0.))
        #     pc_in = rotate_forward(pc_in, R, T)

        if self.split == 'train':
            max_angle = self.max_r
            rotz = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            roty = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            rotx = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            transl_x = np.random.uniform(-self.max_t, self.max_t)
            transl_y = np.random.uniform(-self.max_t, self.max_t)
            transl_z = np.random.uniform(-self.max_t, self.max_t)
            # transl_z = np.random.uniform(-self.max_t, min(self.max_t, 1.))
            initial_RT = 0
        else:
            initial_RT = self.val_RT[idx]
            rotz = initial_RT[6]
            roty = initial_RT[5]
            rotx = initial_RT[4]
            transl_x = initial_RT[1]
            transl_y = initial_RT[2]
            transl_z = initial_RT[3]

        # 随机设置一定范围内的标定参数扰动值
        # train的时候每次都随机生成,每个epoch使用不同的参数
        # test则在初始化的时候提前设置好,每个epoch都使用相同的参数
        # R = mathutils.Euler((rotx, roty, rotz), 'XYZ')
        R = mathutils.Euler((rotx, roty, rotz))
        T = mathutils.Vector((transl_x, transl_y, transl_z))

        R, T = invert_pose(R, T)
        R, T = torch.tensor(R), torch.tensor(T)

        #io.imshow(depth_img.numpy(), cmap='jet')
        #io.show()
        calib = calib_cam02
        # calib = get_calib_kitti_odom(int(seq))
        if h_mirror:
            calib[2] = (img.shape[2] / 2)*2 - calib[2]

        # sample = {'rgb': img, 'point_cloud': pc_in, 'calib': calib, 'pc_org': pc_org, 'img_path': img_path,
        #           'tr_error': T, 'rot_error': R, 'seq': int(seq), 'rgb_name': rgb_name, 'item': item,
        #           'extrin': E_RT, 'initial_RT': initial_RT}
        sample = {'rgb': img, 'point_cloud': pc_in, 'calib': calib, 'pc_org': pc_org, 'img_path': img_path,
                  'tr_error': T, 'rot_error': R, 'rgb_name': rgb_name + '.png', 'item': item,
                  'extrin': E_RT, 'initial_RT': initial_RT, 'pc_lidar': pc_lidar}

        return sample

class DatasetLidarCameraNuscenes(Dataset):

    def __init__(self, dataset_dir, transform=None, augmentation=False, use_reflectance=False,
                 max_t=1.5, max_r=15.0, split='val', device='cpu', val_sequence='2011_09_29_drive_0004_sync'):
        
        super(DatasetLidarCameraNuscenes, self).__init__()
        self.max_r = max_r
        self.max_t = max_t
        self.root_dir = dataset_dir #/mnt/sjmoon/nuscenes
        self.split = split

        # NuScenes 데이터셋을 로드합니다.
        self.nusc = NuScenes(version='v1.0-mini', dataroot=self.root_dir, verbose=True)

        self.val_RT = []
        if split == 'val' or split == 'test':
            val_RT_file = os.path.join(dataset_dir,
                                       f'val_RT_seq{val_sequence}_{max_r:.2f}_{max_t:.2f}.csv')
            if os.path.exists(val_RT_file):
                print(f'VAL SET: Using this file: {val_RT_file}')
                df_test_RT = pd.read_csv(val_RT_file, sep=',')
                for index, row in df_test_RT.iterrows():
                    self.val_RT.append(list(row))
            else:
                print(f'TEST SET - Not found: {val_RT_file}')
                print("Generating a new one")
                val_RT_file = open(val_RT_file, 'w')
                val_RT_file = csv.writer(val_RT_file, delimiter=',')
                val_RT_file.writerow(['id', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz'])
                for i in range(len(self.nusc.sample)):
                    rotz = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    roty = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    rotx = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    transl_x = np.random.uniform(-max_t, max_t)
                    transl_y = np.random.uniform(-max_t, max_t)
                    transl_z = np.random.uniform(-max_t, max_t)
                    # transl_z = np.random.uniform(-max_t, min(max_t, 1.))
                    val_RT_file.writerow([i, transl_x, transl_y, transl_z,
                                           rotx, roty, rotz])
                    self.val_RT.append([float(i), transl_x, transl_y, transl_z,
                                         rotx, roty, rotz])

            # assert len(self.val_RT) == len(self.nusc.sample), "Something wrong with test RTs"


    def __len__(self):
        return len(self.nusc.sample)

    
    def __getitem__(self, idx):
        
        sample = self.nusc.sample[idx]
        real_shape = [900,1600,3]
        
        # Load and transform lidar data.
        lidar_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        lidar_path = self.nusc.get_sample_data_path(sample['data']['LIDAR_TOP'])
        # lidar_points = LidarPointCloud.from_file(lidar_path)
        lidar_points = LidarPointCloud.from_file(lidar_path)
        # valid_indices = lidar_points.points[:, 0] < -3.
        # valid_indices = valid_indices | (lidar_points.points[:, 0] > 3.)
        # valid_indices = valid_indices | (lidar_points.points[:, 1] < -3.)
        # valid_indices = valid_indices | (lidar_points.points[:, 1] > 3.)
        # lidar_points.points[valid_indices]

        # 카메라 데이터를 로드합니다.
        cam_data = self.nusc.get('sample_data', sample['data']['CAM_FRONT'])
        cam_path = self.nusc.get_sample_data_path(sample['data']['CAM_FRONT'])
        im = cv2.imread(cam_path)
        img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        # img = Image.open(cam_path)

        # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
        # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
        lidar_cs_record = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        lidar_points.rotate(Quaternion(lidar_cs_record['rotation']).rotation_matrix)
        lidar_points.translate(np.array(lidar_cs_record['translation']))

        # Second step: transform from ego to the global frame.
        lidar_poserecord = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
        lidar_points.rotate(Quaternion(lidar_poserecord['rotation']).rotation_matrix)
        lidar_points.translate(np.array(lidar_poserecord['translation']))

        # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
        poserecord = self.nusc.get('ego_pose', cam_data['ego_pose_token'])
        lidar_points.translate(-np.array(poserecord['translation']))
        lidar_points.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

        # Fourth step: transform from ego into the camera.
        cs_record = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        lidar_points.translate(-np.array(cs_record['translation']))
        lidar_points.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)
        transform_points =lidar_points.points

        # Fifth step: actually take a "picture" of the point cloud.
        # Grab the depths (camera frame z axis points away from the camera).
        depths = lidar_points.points[2, :]

        # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
        points = view_points(lidar_points.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)
        # points = view_points(lidar_points.points, np.array(cs_record['camera_intrinsic']), normalize=True)

        # 투영된 점들이 이미지 안에 있는지 확인하고, 그렇다면 그 위치에 따라 점들을 그립니다.
        mask = np.ones(depths.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > 0)
        mask = np.logical_and(mask, points[0, :] > 1)
        mask = np.logical_and(mask, points[0, :] < real_shape[1]) # 이미지의 width 안에 있는지 확인
        mask = np.logical_and(mask, points[1, :] > 1)
        mask = np.logical_and(mask, points[1, :] < real_shape[0]) # 이미지의 height 안에 있는지 확인

        # 마스킹된 Lidar 포인트들을 표시합니다.
        # lidar_points_np = points[:, mask]
        depths = depths[mask]

        lidar_points_np = points

        # # 이미지 위에 lidar points를 그립니다.
        # plt.figure(figsize=(10, 10))
        # plt.imshow(img)
        # plt.scatter(lidar_points_np[0, :], lidar_points_np[1, :], c=lidar_points_np[2, :], s=2)
        # plt.axis('off')
        # plt.show()

        lidar_points_torch = torch.from_numpy(transform_points)
        disp_lidar_points = torch.from_numpy(lidar_points_np)

        # extrinsic parameter error calculation
        if self.split == 'train':
            max_angle = self.max_r
            rotz = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            roty = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            rotx = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            transl_x = np.random.uniform(-self.max_t, self.max_t)
            transl_y = np.random.uniform(-self.max_t, self.max_t)
            transl_z = np.random.uniform(-self.max_t, self.max_t)
            # transl_z = np.random.uniform(-self.max_t, min(self.max_t, 1.))
            initial_RT = 0
        else:
            initial_RT = self.val_RT[idx]
            rotz = initial_RT[6]
            roty = initial_RT[5]
            rotx = initial_RT[4]
            transl_x = initial_RT[1]
            transl_y = initial_RT[2]
            transl_z = initial_RT[3]

        R = mathutils.Euler((rotx, roty, rotz))
        T = mathutils.Vector((transl_x, transl_y, transl_z))

        R, T = invert_pose(R, T)
        R, T = torch.tensor(R), torch.tensor(T)

        calib = np.array(cs_record['camera_intrinsic'])

        # sample1 = { 'rgb': img, 'point_cloud': lidar_points, 'calib': calib,
        #           'tr_error': T, 'rot_error': R, 'initial_RT': initial_RT  , 'cs_record' : lidar_cs_record , 'pose_record': lidar_poserecord
        #           }
        
        sample1 = { 'rgb': img, 'point_cloud': lidar_points, 'calib': calib,
                  'tr_error': T, 'rot_error': R, 'initial_RT': initial_RT
                  }

        return sample1


# In[5]:


# import matplotlib
# matplotlib.rc("font",family='AR PL UMing CN')
plt.rcParams['axes.unicode_minus'] = False
# plt.rc('font',family='Times New Roman')
font_EN = {'family': 'Times New Roman', 'weight': 'normal', 'size': 16}
font_CN = {'family': 'AR PL UMing CN', 'weight': 'normal', 'size': 16}
plt_size = 10.5

ex = Experiment("LCCNet-evaluate-iterative",interactive = True)
ex.captured_out_filter = apply_backspaces_and_linefeeds

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device ='cuda'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# In[6]:


# noinspection PyUnusedLocal
@ex.config
def config():
    dataset = 'kitti/odom'
    # data_folder = '/mnt/data/kitti_odometry'
    # data_folder = "/data/kitti/kitti_odometry" # sapeon server gpu a100
    # data_folder = "/data/kitti/raw_data" # sapeon server gpu a100
    # data_folder = "/mnt/sjmoon/kitti/raw_data" # sapeon desktop gpu 4090 rawdata
    data_folder = "/mnt/sjmoon/nuscenes" # sapeon desktop gpu 4090 nuscenes data
    # test_sequence = 0
    test_sequence=      '2011_09_30_drive_0016_sync'   #  '2011_09_29_drive_0004_sync'    '2011_09_26_drive_0005_sync' 
    use_prev_output = False
    max_t = 0.25
    max_r = 10.0
    occlusion_kernel = 5
    occlusion_threshold = 3.0
    network = 'Res_f1'
    norm = 'bn'
    show =  False
    use_reflectance = False
    weight = None  # List of weights' path, for iterative refinement
    # weight =  './checkpoints/kitti/odom/val_seq_07/models/checkpoint_r7.50_t0.20_e110_0.383.tar'
    save_name = None
    # Set to True only if you use two network, the first for rotation and the second for translation
    rot_transl_separated = False
    random_initial_pose = False
    save_log = False
    dropout = 0.0
    max_depth = 80.
    iterative_method = 'single' # ['multi_range', 'single_range', 'single']
    output = './output'
    save_image = False
    outlier_filter = True
    outlier_filter_th = 10
    out_fig_lg = 'EN' # [EN, CN]
    dense_resoltuion = 2
    num_kp = 300
    dataset_type ='nuscenes'

weights = [
    './checkpoints/kitti/odom/val_seq_2011_09_26_drive_0005_sync/models/checkpoint_r10.00_t0.25_e248_1.889.tar',
   './checkpoints/kitti/odom/val_seq_00/models/checkpoint_r10.00_t0.25_e4_22.218.tar',
   './checkpoints/kitti/odom/val_seq_07/models/checkpoint_r7.50_t0.20_e110_0.383.tar',
   './checkpoints/kitti/odom/val_seq_07/models/checkpoint_r7.50_t0.20_e110_0.383.tar',
   './checkpoints/kitti/odom/val_seq_07/models/checkpoint_r7.50_t0.20_e110_0.383.tar',
  # './checkpoints/kitti/odom/val_seq_07/models/checkpoint_r7.50_t0.20_e110_0.383.tar',
]

"""
weights = [
   './pretrained/kitti/kitti_iter1.tar',
   './pretrained/kitti/kitti_iter2.tar',
   './pretrained/kitti/kitti_iter3.tar',
   './pretrained/kitti/kitti_iter4.tar',
   './pretrained/kitti/kitti_iter5.tar',
]
"""

EPOCH = 1


def _init_fn(worker_id, seed):
    seed = seed + worker_id + EPOCH * 100
    print(f"Init worker {worker_id} with seed {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def two_images_side_by_side(img_a, img_b):
    assert img_a.shape == img_b.shape, f'{img_a.shape} vs {img_b.shape}'
    assert img_a.dtype == img_b.dtype
    b, h, w, c = img_a.shape
    canvas = np.zeros((b, h, 2 * w, c), dtype=img_a.cpu().numpy().dtype)
    canvas[:, :, 0 * w:1 * w, :] = img_a.cpu().numpy()
    canvas[:, :, 1 * w:2 * w, :] = img_b.cpu().numpy()
    #canvas[:, :, : , 0 * w:1 * w] = img_a.cpu().numpy()
    #canvas[:, :, : , 1 * w:2 * w] = img_b.cpu().numpy()
    return canvas

def get_2D_lidar_projection(pcl, cam_intrinsic):
    pcl_xyz = cam_intrinsic @ pcl.T
    pcl_xyz = pcl_xyz.T
    pcl_z = pcl_xyz[:, 2]
    pcl_xyz = pcl_xyz / (pcl_xyz[:, 2, None] + 1e-10)
    pcl_uv = pcl_xyz[:, :2]

    return pcl_uv, pcl_z


def lidar_project_depth(pc_rotated, cam_calib, img_shape):
    # pc_rotated = pc_rotated[:3, :].detach().cpu().numpy()
    # cam_intrinsic = cam_calib.numpy()
    # pcl_uv, pcl_z = get_2D_lidar_projection(pc_rotated.T, cam_intrinsic)
    
    pc_rotated = pc_rotated[:3, :].detach()
    cam_intrinsic = torch.tensor(cam_calib, dtype=torch.float32).cuda()
    pcl_uv, pcl_z = get_2D_lidar_projection(pc_rotated.t(), cam_intrinsic)
    mask = (pcl_uv[:, 0] > 0) & (pcl_uv[:, 0] < img_shape[1]) & (pcl_uv[:, 1] > 0) & (
            pcl_uv[:, 1] < img_shape[0] ) & (pcl_z > 0)
    
    # mask1 = (pcl_uv[:, 1] < 188)

    pcl_uv_no_mask = pcl_uv
    pcl_uv = pcl_uv[mask]    
    pcl_z = pcl_z[mask]
    pcl_uv = torch.tensor(pcl_uv, dtype=torch.int32).cuda()
    pcl_z = torch.tensor(pcl_z, dtype=torch.float32).cuda()
    pcl_z = pcl_z.reshape(-1, 1)

    depth_img = np.zeros((img_shape[0], img_shape[1], 1))
    depth_img = torch.from_numpy(depth_img.astype(np.float32)).cuda()
    depth_img[pcl_uv[:, 1], pcl_uv[:, 0]] = pcl_z  
    depth_img = depth_img.permute(2, 0, 1)
    points_index = torch.arange(pcl_uv_no_mask.shape[0], device='cuda')[mask]

    return depth_img, pcl_uv , pcl_z , points_index  


def dense_map(Pts ,n, m, grid):
    ng = 2 * grid + 1

    # mX = np.zeros((m,n)) + np.float("inf")
    # mY = np.zeros((m,n)) + np.float("inf")
    # mD = np.zeros((m,n))

    # mX[np.int32(Pts[1]),np.int32(Pts[0])] = Pts[0] - np.round(Pts[0])
    # mY[np.int32(Pts[1]),np.int32(Pts[0])] = Pts[1] - np.round(Pts[1])
    # mD[np.int32(Pts[1]),np.int32(Pts[1])] = Pts[2]

    # KmX = np.zeros((ng, ng, m - ng, n - ng))
    # KmY = np.zeros((ng, ng, m - ng, n - ng))
    # KmD = np.zeros((ng, ng, m - ng, n - ng))

    mX = torch.full((m, n), float('inf'), dtype=torch.float32, device='cuda')
    mY = torch.full((m, n), float('inf'), dtype=torch.float32, device='cuda')
    mD = torch.zeros((m, n), dtype=torch.float32, device='cuda')

    mX_idx = torch.tensor(Pts[1], dtype=torch.int64)
    mY_idx = torch.tensor(Pts[0], dtype=torch.int64)

    mX[mX_idx, mY_idx] = Pts[0] - torch.round(Pts[0])
    mY[mX_idx, mY_idx] = Pts[1] - torch.round(Pts[1])
    mD[mX_idx, mY_idx] = Pts[2]

    KmX = torch.zeros((ng, ng, m - ng, n - ng), dtype=torch.float32, device='cuda')
    KmY = torch.zeros((ng, ng, m - ng, n - ng), dtype=torch.float32, device='cuda')
    KmD = torch.zeros((ng, ng, m - ng, n - ng), dtype=torch.float32, device='cuda')

    for i in range(ng):
        for j in range(ng):
            KmX[i,j] = mX[i : (m - ng + i), j : (n - ng + j)] - grid - 1 +i
            KmY[i,j] = mY[i : (m - ng + i), j : (n - ng + j)] - grid - 1 +i
            KmD[i,j] = mD[i : (m - ng + i), j : (n - ng + j)]
    # S = np.zeros_like(KmD[0,0])
    # Y = np.zeros_like(KmD[0,0])
    S = torch.zeros_like(KmD[0, 0])
    Y = torch.zeros_like(KmD[0, 0])

    for i in range(ng):
        for j in range(ng):
            # s = 1/np.sqrt(KmX[i,j] * KmX[i,j] + KmY[i,j] * KmY[i,j])
            s = 1 / torch.sqrt(KmX[i, j] * KmX[i, j] + KmY[i, j] * KmY[i, j])
            Y = Y + s * KmD[i,j]
            S = S + s

    S[S == 0] = 1
    # out = np.zeros((m,n))
    out = torch.zeros((m, n), dtype=torch.float32, device='cuda')
    out[grid + 1 : -grid, grid + 1 : -grid] = Y/S
    return out

def colormap(disp):
    """"Color mapping for disp -- [H, W] -> [3, H, W]"""
    disp_np = disp.cpu().numpy()        # tensor -> numpy
    # disp_np = disp
    # vmax = np.percentile(disp_np, 95)
    vmin = disp_np.min()
    vmax = disp_np.max()
    normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')  #magma, plasma, etc.
    colormapped_im = (mapper.to_rgba(disp_np)[:, :, :3])
    # return colormapped_im.transpose(2, 0, 1)
    colormapped_tensor = torch.from_numpy(colormapped_im).permute(2, 0, 1).to(dtype=torch.float32).cuda()
    # colormapped_tensor = torch.from_numpy(colormapped_im).
    return colormapped_tensor

def farthest_point_sampling(points, k):
    """
    Args:
        points (torch.Tensor): (N, 3) shape의 포인트 집합
        k (int): 선택할 중심 포인트의 개수
    Returns:
        torch.Tensor: (k, 3) shape의 선택된 중심 포인트 좌표
        torch.Tensor: (k) shape의 선택된 중심 포인트 인덱스
    """
    N, _ = points.shape
    centroids = torch.zeros(k, dtype=torch.long, device=points.device)
    distance = torch.ones(N, device=points.device) * 1e10

    # 첫 번째 중심 포인트를 무작위로 선택
    farthest = torch.randint(0, N, (1,), dtype=torch.long, device=points.device)

    for i in range(k):
        # 가장 먼 지점을 중심 포인트로 선택
        centroids[i] = farthest
        centroid = points[farthest, :].view(1, 3)

        # 선택한 중심 포인트와 다른 모든 포인트 간의 거리 계산
        dist = torch.sum((points - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]

        # 가장 먼 포인트를 찾는다
        farthest = torch.max(distance, dim=0)[1]

    # 선택된 중심 포인트 좌표 및 인덱스 반환
    return centroids ,points[centroids]

def knn(x, y ,k):
# #         print (" x shape = " , x.shape)
#         inner = -2*torch.matmul(x.transpose(-2, 1), x)
#         xx = torch.sum(x**2, dim=1, keepdim=True)
# #         print (" xx shape = " , x.shape)
#         pairwise_distance = -xx - inner - xx.transpose(4, 1)
    # mask_x = (x[: , 2] > 0.5) & (x[: , 2] < 0.8)
    # mask_y = (y[: , 2] > 0.5) & (y[: , 2] < 0.8)
    # x1 = x[mask_x]
    # y1 = y[mask_y]
    # mask_x1= np.in1d(mask_x,mask_y)
    # mask_y1= np.in1d(mask_y,mask_x)
    # x2 = x[mask_x1]
    # y2 = y[mask_y1]
    # x2 = torch.from_numpy(x2)  # NumPy 배열을 PyTorch Tensor로 변환
    # y2 = torch.from_numpy(y2)  # NumPy 배열을 PyTorch Tensor로 변환
    # pairwise_distance = F.pairwise_distance(x,y)
    
    # #### monitoring x/y point #####################
    # print ("x2 x_point min =" , torch.min(x[:,0]))
    # print ("x2 x_point max =" , torch.max(x[:,0]))
    # print ("y2 x_point min =" , torch.min(y[:,0]))
    # print ("y2 x_point max =" , torch.max(y[:,0]))
    # print ("x2 y_point min =" , torch.min(x[:,1]))
    # print ("x2 y_point max =" , torch.max(x[:,1]))
    # print ("y2 y_point min =" , torch.min(y[:,1]))
    # print ("y2 y_point max =" , torch.max(y[:,1]))
    # print ("x2 depth min =" , torch.min(x[:,2]))
    # print ("x2 depth max =" , torch.max(x[:,2]))
    # print ("y2 depth min =" , torch.min(y[:,2]))
    # print ("y2 depth max =" , torch.max(y[:,2]))
    # ##############################################
    
    # 일정 depth range (min_depth, max_depth)
    min_depth = 0.05
    max_depth = 0.2
    
    # y[:, 2] = 1 - y[:, 2] # 세 번째 열 값 반전
    # min_depth <= depth <= max_depth 인 point들의 인덱스를 구합니다.
    depth_mask1 = (x[:, 2] >= min_depth) & (x[:, 2] <= max_depth) # & (x[:,1] >= 0.6 )
    depth_mask2 = (y[:, 2] >= min_depth) & (y[:, 2] <= max_depth) # & (y[:,1] >= 0.6 )
    # depth_indices1 = np.where(depth_mask1)[0]
    # depth_indices2 = np.where(depth_mask2)[0]
    depth_indices1 = torch.nonzero(depth_mask1).squeeze()
    depth_indices2 = torch.nonzero(depth_mask2).squeeze()

    x1 = x[depth_indices1]
    y1 = y[depth_indices2]

    # mask_x1= np.in1d(depth_indices1,depth_indices2)
    # mask_y1= np.in1d(depth_indices2,depth_indices1)
    mask_x1 = (depth_indices1.view(-1, 1)== depth_indices2.view(1, -1)).any(dim=1)
    mask_y1 = (depth_indices2.view(-1, 1) == depth_indices1.view(1, -1)).any(dim=1)
    # mask_x1 = torch.tensor([elem in depth_indices2.cpu().numpy() for elem in depth_indices1.cpu().numpy()], device=x.device, dtype=torch.bool)
    # mask_y1 = torch.tensor([elem in depth_indices1.cpu().numpy() for elem in depth_indices2.cpu().numpy()], device=y.device, dtype=torch.bool)

    x2 = x1.index_select(0, torch.nonzero(mask_x1).squeeze())
    y2 = y1.index_select(0, torch.nonzero(mask_y1).squeeze())
    # x2 = x1[mask_x1]
    # y2 = y1[mask_y1]
    
    if x2.shape[0] <= k :
        # x2 = torch.zeros(k, 3 , device=x.device)
        # y2 = torch.zeros(k, 3,  device=y.device)
        ### 부족하면 무조건 랜덤 수 채우기
        x2 = torch.rand(k, 3).cuda()
        y2 = torch.rand(k, 3).cuda()
            
  
    #### 유사한 포인트 뽑기 using KNN #####
    pairwise_distance = F.pairwise_distance(x2, y2)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    # top_indices = torch.topk(pairwise_distance.flatten(), k=k, largest=False)
    # top_indices = top_indices.indices
    # indices = np.unravel_index(top_indices, pairwise_distance.shape)
    # top_indices = np.asarray(top_indices).T
    
    #### 가장 먼 포인트 들 뽑기 #########
    # idx ,_ = farthest_point_sampling(x2,k)

    top_x = x2[idx]
    top_y = y2[idx]
    # print ("x point of z =" , top_x[3])
    # print ("y point of z =" , top_y[3])
    # top_y[:, 2] =  1- top_y[:, 2] # 세 번째 열의 값에서 1을 빼기 
    # print ("y point of rev z =" , top_y[3])
    
    corrs = torch.cat([top_x,top_y] ,dim=1) 
        
    return idx , corrs


def corr_gen_withZ( gt_points_index, points_index, gt_uv, uv , gt_z, z, origin_img_shape, resized_shape, num_kp = 500) :
    
    #only numpy operation
    # inter_gt_uv_mask = np.in1d(gt_points_index , points_index)
    # inter_uv_mask    = np.in1d(points_index , gt_points_index)

    inter_gt_uv_mask = torch.tensor(np.in1d(gt_points_index.cpu().numpy(), points_index.cpu().numpy()), device='cuda')
    inter_uv_mask = torch.tensor(np.in1d(points_index.cpu().numpy(), gt_points_index.cpu().numpy()), device='cuda')
    gt_uv = gt_uv[inter_gt_uv_mask]
    uv    = uv[inter_uv_mask] 
    gt_z = gt_z[inter_gt_uv_mask]
    z    = z[inter_uv_mask] 
    # gt_uvz = np.concatenate([gt_uv,gt_z], axis=1)
    # uvz= np.concatenate([uv,z],axis=1)
    # corrs = np.concatenate([gt_uvz, uvz], axis=1)
    # corrs = torch.tensor(corrs)
    gt_uvz = torch.cat([gt_uv, gt_z], dim=1)
    uvz = torch.cat([uv, z], dim=1)
    corrs = torch.cat([gt_uvz, uvz], dim=1)

    # gt_points = torch.tensor(gt_uvz)
    # target_points = torch.tensor(uvz)
    # scale_img = np.array (resized_shape) / np.array(origin_img_shape) 
    
    # #### monitoring x/y point #####################
    # print ("origin gt x_point min =" ,     torch.min(corrs[:,0]))
    # print ("origin gt x_point max =" ,     torch.max(corrs[:,0]))
    # print ("origin target x_point min =" , torch.min(corrs[:,3]))
    # print ("origin target x_point max =" , torch.max(corrs[:,3]))
    # print ("origin gt y_point min =" ,     torch.min(corrs[:,1]))
    # print ("origin gt y_point max =" ,     torch.max(corrs[:,1]))
    # print ("origin target y_point min =" , torch.min(corrs[:,1]))
    # print ("origin target y_point max =" , torch.max(corrs[:,1]))
    # print ("origin gt depth min =" ,       torch.min(corrs[:,2]))
    # print ("origin gt depth max =" ,       torch.max(corrs[:,2]))
    # print ("origin target depth min =" ,   torch.min(corrs[:,2]))
    # print ("origin target depth max =" ,   torch.max(corrs[:,2]))
    # ##############################################
    
    # corrs[:, 0] = (0.5*corrs[:, 0])/1280
    corrs[:, 0] = corrs[:, 0]/origin_img_shape[1] 
    # corrs[:, 1] = (0.5*corrs[:, 1])/384
    corrs[:, 1] = corrs[:, 1]/origin_img_shape[0] 
    if corrs[:, 2].numel() > 0:
        corrs[:, 2] = (corrs[:, 2]-torch.min(corrs[:, 2]))/(torch.max(corrs[:, 2]) - torch.min(corrs[:, 2]))
    else :
        corrs[:, 2] = (corrs[:, 2]-0)/(80 - 0)
    # corrs[:, 3] = (0.5*corrs[:, 3])/1280 + 0.5
    corrs[:, 3] = corrs[:, 3]/origin_img_shape[1]         
    # corrs[:, 4] = (0.5*corrs[:, 4])/384
    corrs[:, 4] = corrs[:, 4]/origin_img_shape[0]
    if corrs[:, 5].numel() > 0:
        corrs[:, 5] = (corrs[:, 5]-torch.min(corrs[:, 5]))/(torch.max(corrs[:, 5]) - torch.min(corrs[:, 5])) 
    else :
        corrs[:, 5] = (corrs[:, 5]-0)/(80 - 0)

    # #### monitoring x/y point #####################
    # print ("normalized gt x_point min =" ,     torch.min(corrs[:,0]))
    # print ("normalized gt x_point max =" ,     torch.max(corrs[:,0]))
    # print ("normalized target x_point min =" , torch.min(corrs[:,3]))
    # print ("normalized target x_point max =" , torch.max(corrs[:,3]))
    # print ("normalized gt y_point min =" ,     torch.min(corrs[:,1]))
    # print ("normalized gt y_point max =" ,     torch.max(corrs[:,1]))
    # print ("normalized target y_point min =" , torch.min(corrs[:,1]))
    # print ("normalized target y_point max =" , torch.max(corrs[:,1]))
    # print ("normalized gt depth min =" ,       torch.min(corrs[:,2]))
    # print ("normalized gt depth max =" ,       torch.max(corrs[:,2]))
    # print ("normalized target depth min =" ,   torch.min(corrs[:,2]))
    # print ("normalized target depth max =" ,   torch.max(corrs[:,2]))
    # ##############################################

    if corrs.shape[0] <= num_kp :
        # corrs = torch.zeros(num_kp, 6)
        diff = num_kp - corrs.shape[0]
        rand_values = torch.randn(diff, 6).cuda()
        corrs = torch.cat([corrs, rand_values], dim=0)
        # target_points = torch.zeros(num_kp, 3)
        # corrs[:, 2] = corrs[:, 2] + 0.5 # for only uv matching
        # corrs[:, 3] = corrs[:, 3] + 0.5 # for uvz matching

    corrs_knn_idx ,corrs_prev = knn(corrs[:,:3], corrs[:,3:], num_kp) # knn 2d point-cloud trim

    corrs = corrs[corrs_knn_idx]   
    corrs1 = corrs_prev
    # corrs = corrs[z_mask]    
    # corrs = torch.cat([top_gt_points,top_target_points],dim=1)

    # assert (0.0 <= corrs[:, 0]).all() and (corrs[:, 0] <= 0.5).all()
    # assert (0.0 <= corrs[:, 1]).all() and (corrs[:, 1] <= 1.0).all()
    # assert (0.0 <= corrs[:, 2]).all() and (corrs[:, 2] <= 1.0).all()
    # assert (0.5 <= corrs[:, 3]).all() and (corrs[:, 3] <= 1.0).all()
    # assert (0.0 <= corrs[:, 4]).all() and (corrs[:, 4] <= 1.0).all()
    # assert (0.0 <= corrs[:, 5]).all() and (corrs[:, 5] <= 1.0).all()
    
    return corrs1

def reduce_lidar_resolution(original_lidar_data, original_line_num=64, target_line_num=32):
    assert original_line_num % target_line_num == 0
    ratio = original_line_num // target_line_num

    sorted_lidar_data = original_lidar_data[original_lidar_data[:, -1].argsort()]
    data_per_line = len(sorted_lidar_data) // original_line_num
    target_lidar_data = []

    for i in range(0, original_line_num, ratio):
        target_lidar_data.append(sorted_lidar_data[i * data_per_line:(i + ratio) * data_per_line])

    reduced_lidar_data = np.vstack(target_lidar_data)
    return reduced_lidar_data

def random_mask(sbs_img,grid_size=(32, 32), mask_value=0):
    # sbs_img shape: [batch, channel, height, width]
    batch_size, _, height, width = sbs_img.shape
    mask = torch.ones_like(sbs_img)

    grid_height, grid_width = grid_size

    for i in range(height // grid_height):
        for j in range(width // grid_width):
            if torch.rand(1) > 0.75:  # Randomly choose whether to mask this grid or not
                # Apply the mask to the corresponding area in the image
                mask[:, :, i*grid_height:(i+1)*grid_height,j*grid_width:(j+1)*grid_width] = mask_value

    return sbs_img * mask

def lidar_project_depth_nuscenes(nusc,lidar_points, cam_data, real_shape):
    
    cs_record = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    pc_points = lidar_points.cpu().numpy()
    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = pc_points[2, :]

    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    points = view_points(pc_points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)
    # points = view_points(lidar_points.points, np.array(cs_record['camera_intrinsic']), normalize=True)

    # 투영된 점들이 이미지 안에 있는지 확인하고, 그렇다면 그 위치에 따라 점들을 그립니다.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > 0)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < real_shape[1]) # 이미지의 width 안에 있는지 확인
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < real_shape[0]) # 이미지의 height 안에 있는지 확인

    pcl_uv = points.transpose(1,0)[:,:2]
    pcl_z = points.transpose(1,0)[:,2]
    pcl_uv_no_mask = pcl_uv
    pcl_uv = pcl_uv[mask]    
    # pcl_z = pcl_z[mask]
    pcl_z = depths[mask]
    pcl_uv = torch.tensor(pcl_uv, dtype=torch.int32).cuda()
    pcl_z = torch.tensor(pcl_z, dtype=torch.float32).cuda()
    pcl_z = pcl_z.reshape(-1, 1)
    
    depth_img = np.zeros((real_shape[0], real_shape[1], 1))
    depth_img = torch.from_numpy(depth_img.astype(np.float32)).cuda()
    depth_img[pcl_uv[:, 1], pcl_uv[:, 0]] = pcl_z  
    depth_img = depth_img.permute(2, 0, 1)
    points_index = torch.arange(pcl_uv_no_mask.shape[0], device='cuda')[mask]

    return depth_img, pcl_uv , pcl_z , points_index

def transform_gt(nusc, lidar_data, lidar_points ,cam_data) :
    # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
    # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
    lidar_cs_record = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
    lidar_points.rotate(Quaternion(lidar_cs_record['rotation']).rotation_matrix)
    lidar_points.translate(np.array(lidar_cs_record['translation']))

    # Second step: transform from ego to the global frame.
    lidar_poserecord = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    lidar_points.rotate(Quaternion(lidar_poserecord['rotation']).rotation_matrix)
    lidar_points.translate(np.array(lidar_poserecord['translation']))

    # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
    poserecord = nusc.get('ego_pose', cam_data['ego_pose_token'])
    lidar_points.translate(-np.array(poserecord['translation']))
    lidar_points.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    cs_record = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    lidar_points.translate(-np.array(cs_record['translation']))
    lidar_points.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

    return lidar_points

def maintain_aspect_ratio_resize(img, size):
    batch_size, channels, h, w = img.shape
    target_h, target_w = size

    if (float(w)/h) > (float(target_w)/target_h):
        # 원본 이미지의 가로세로 비율이 목표 크기의 가로세로 비율보다 큰 경우
        # 가로 길이를 목표 크기에 맞춥니다.
        new_w = target_w
        new_h = int(h * new_w / w)
    else:
        # 그 외의 경우 세로 길이를 목표 크기에 맞춥니다.
        new_h = target_h
        new_w = int(w * new_h / h)

    resized_img = F.interpolate(img, size=[new_h, new_w], mode="bilinear")

    # 만약 새로운 이미지의 한 축의 길이가 목표 크기보다 작다면,
    # 나머지 공간을 0으로 채워줍니다.
    if new_h < target_h:
        diff = target_h - new_h
        padding = torch.zeros((batch_size, channels, diff, new_w))
        resized_img = torch.cat((resized_img, padding), dim=2)
    elif new_w < target_w:
        diff = target_w - new_w
        padding = torch.zeros((batch_size, channels, new_h, diff))
        resized_img = torch.cat((resized_img, padding), dim=3)

    return resized_img

# In[7]:


#@ex.automain
@ex.main
def main(_config, seed):
    global EPOCH, weights
    if _config['weight'] is not None:
        weights = _config['weight']

    if _config['iterative_method'] == 'single':
        weights = [weights[0]]


    input_size = (256, 512)

    # split = 'test'
    if _config['random_initial_pose']:
        split = 'test_random'

    if _config['test_sequence'] is None:
        raise TypeError('test_sequences cannot be None')
    else:
        if isinstance(_config['test_sequence'], int):
            _config['test_sequence'] = f"{_config['test_sequence']:02d}"
        
    if _config['dataset_type'] == 'kitti' :
        img_shape = (384, 1280) #for kitti
        # dataset_class = DatasetLidarCameraKittiOdometry
        # dataset_class = DatasetTest # LCCNet test dataset
        dataset_class = DatasetLidarCameraKittiRaw # Kitti raw dataset
        dataset_val = dataset_class(_config['data_folder'], max_r=_config['max_r'], max_t=_config['max_t'],
                                    split='val', use_reflectance=_config['use_reflectance'],
                                    val_sequence=_config['test_sequence']) #for kitti data instance
        # dataset_val = dataset_class(_config['data_folder'], max_r=_config['max_r'], max_t=_config['max_t'],
        #                             split='random', use_reflectance=_config['use_reflectance'],
        #                             test_sequence=_config['test_sequence'], est='rot')
    elif _config['dataset_type'] == 'nuscenes' :
        img_shape = (900, 1600) #for nuscenes 
        # NuScenes 데이터셋을 로드합니다.
        nusc = NuScenes(version='v1.0-mini', dataroot=_config['data_folder'], verbose=True)
        dataset_class = DatasetLidarCameraNuscenes # Nuscenes dataset
        dataset_val = dataset_class(_config['data_folder'], max_r=_config['max_r'], max_t=_config['max_t'],
                                    split='val', val_sequence=_config['test_sequence']) # for nuscenes data instance

    np.random.seed(seed)
    torch.random.manual_seed(seed)

    def init_fn(x):
        return _init_fn(x, seed)

    num_worker = 6
    batch_size = 10

    TestImgLoader = torch.utils.data.DataLoader(dataset=dataset_val,
                                                shuffle=False,
                                                batch_size=batch_size,
                                                num_workers=num_worker,
                                                worker_init_fn=init_fn,
                                                collate_fn=merge_inputs,
                                                drop_last=True,
                                                pin_memory=False)

    print(len(TestImgLoader))

    models = [] # iterative model
    # MonoDelsNet_model = MonoDelsNet()

    for i in range(len(weights)):
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
            model = DepthCalibTranformer(input_size, use_feat_from=feat, md=md,
                             use_reflectance=_config['use_reflectance'], dropout=_config['dropout'], num_kp = _config["num_kp"])
        else:
            raise TypeError("Network unknown")

        print ("weight[i]", weights[i])
        checkpoint = torch.load(weights[i], map_location='cpu')
        saved_state_dict = checkpoint['state_dict']
        model.load_state_dict(saved_state_dict)
        model = model.cuda()
        model.eval()
        models.append(model)

    if _config['save_log']:
        log_file = f'./results_for_paper/log_seq{_config["test_sequence"]}.csv'
        log_file = open(log_file, 'w')
        log_file = csv.writer(log_file)
        header = ['frame']
        for i in range(len(weights) + 1):
            header += [f'iter{i}_error_t', f'iter{i}_error_r', f'iter{i}_error_x', f'iter{i}_error_y',
                       f'iter{i}_error_z', f'iter{i}_error_r', f'iter{i}_error_p', f'iter{i}_error_y']
        log_file.writerow(header)

    show = _config['show']
    # save image to the output path
    _config['output'] = os.path.join(_config['output'], _config['iterative_method'])
    rgb_path = os.path.join(_config['output'], 'rgb')
    if not os.path.exists(rgb_path):
        os.makedirs(rgb_path)
    depth_path = os.path.join(_config['output'], 'depth')
    if not os.path.exists(depth_path):
        os.makedirs(depth_path)
    input_path = os.path.join(_config['output'], 'input')
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    gt_path = os.path.join(_config['output'], 'gt')
    if not os.path.exists(gt_path):
        os.makedirs(gt_path)
    if _config['out_fig_lg'] == 'EN':
        results_path = os.path.join(_config['output'], 'results_en')
    elif _config['out_fig_lg'] == 'CN':
        results_path = os.path.join(_config['output'], 'results_cn')
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    pred_path = os.path.join(_config['output'], 'pred')
    for it in range(len(weights)):
        if not os.path.exists(os.path.join(pred_path, 'iteration_'+str(it+1))):
            os.makedirs(os.path.join(pred_path, 'iteration_'+str(it+1)))

    # save pointcloud to the output path
    pc_lidar_path = os.path.join(_config['output'], 'pointcloud', 'lidar')
    if not os.path.exists(pc_lidar_path):
        os.makedirs(pc_lidar_path)
    pc_input_path = os.path.join(_config['output'], 'pointcloud', 'input')
    if not os.path.exists(pc_input_path):
        os.makedirs(pc_input_path)
    pc_pred_path = os.path.join(_config['output'], 'pointcloud', 'pred')
    if not os.path.exists(pc_pred_path):
        os.makedirs(pc_pred_path)


    errors_r = []
    errors_t = []
    errors_t2 = []
    errors_xyz = []
    errors_rpy = []
    all_RTs = []
    mis_calib_list = []
    total_time = 0

    prev_tr_error = None
    prev_rot_error = None

    for i in range(len(weights) + 1):
        errors_r.append([])
        errors_t.append([])
        errors_t2.append([])
        errors_rpy.append([])

    for batch_idx, sample in enumerate(tqdm(TestImgLoader)):
        N = 100 # 500
        # if batch_idx > 200:
        #    break

        log_string = [str(batch_idx)]

        lidar_input = []
        rgb_input = []
        corrs_input = []
        shape_pad_input = []
        pc_rotated_input = []
        RTs = []
        lidar_input_gt =[]

        outlier_filter = False
        rgb_resize_shape = [192,640,3]
        # real_shape = [376 , 1241 ,3]

        if batch_idx == 0 or not _config['use_prev_output']:
            # 프레임을 수정하여 GT를 입력 할 수 있습니다.
            sample['tr_error'] = sample['tr_error'].cuda()
            sample['rot_error'] = sample['rot_error'].cuda()
        else:
            sample['tr_error'] = prev_tr_error
            sample['rot_error'] = prev_rot_error
        
        for idx in range(len(sample['rgb'])):
            
            if _config['dataset_type'] =='kitti':  
                # ProjectPointCloud in RT-pose
                real_shape = [sample['rgb'][idx].shape[0], sample['rgb'][idx].shape[1], sample['rgb'][idx].shape[2]] #[height:width:channel/ 900,1600,3]
                rgb = sample['rgb'][idx]
                rgb = transforms.ToTensor()(rgb).to(device)
      
                pc_lidar = sample['point_cloud'][idx].clone().to(device)

                if _config['max_depth'] < 80.:
                    pc_lidar = pc_lidar[:, pc_lidar[0, :] < _config['max_depth']].clone()

                depth_gt, gt_uv , gt_z , gt_points_index  = lidar_project_depth(pc_lidar, sample['calib'][idx].to(device), real_shape) # image_shape
                depth_gt /= _config['max_depth']  

            elif _config['dataset_type'] =='nuscenes': 
                
                real_shape = [sample['rgb'][idx].shape[0], sample['rgb'][idx].shape[1], sample['rgb'][idx].shape[2]] #[height:width:channel/ 900,1600,3]
                nusc_sample = nusc.sample[idx]
                # Load and transform lidar data.
                lidar_data = nusc.get('sample_data', nusc_sample['data']['LIDAR_TOP'])
                lidar_path = nusc.get_sample_data_path(nusc_sample['data']['LIDAR_TOP'])
                # lidar_points = LidarPointCloud.from_file(lidar_path)
                lidar_points, _ = LidarPointCloud.from_file_multisweep(nusc, nusc_sample, 'LIDAR_TOP', 'LIDAR_TOP', nsweeps=10)
                
                # 카메라 데이터를 로드합니다.
                cam_data = nusc.get('sample_data', nusc_sample['data']['CAM_FRONT'])
                cam_path = nusc.get_sample_data_path(nusc_sample['data']['CAM_FRONT'])
                im = cv2.imread(cam_path)
                img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                rgb = transforms.ToTensor()(img).to(device)

                lidar_points = transform_gt(nusc, lidar_data, lidar_points ,cam_data)

                pc_transformed_gt = torch.from_numpy(lidar_points.points).cuda()
                
                depth_gt, gt_uv , gt_z , gt_points_index  = lidar_project_depth_nuscenes(nusc, pc_transformed_gt, cam_data, real_shape)
                depth_gt /= _config['max_depth']
                
            lidarOnImage = torch.cat((gt_uv, gt_z), dim=1)
            dense_depth_img_gt = dense_map(lidarOnImage.T , real_shape[1], real_shape[0] , _config['dense_resoltuion']) # argument = (lidarOnImage.T , 1241, 376 , 8)
            dense_depth_img_gt = dense_depth_img_gt.to(dtype=torch.uint8)
            dense_depth_img_color_gt = colormap(dense_depth_img_gt)

            # input display
            disp_pc_lidar_np = lidarOnImage.detach().cpu().numpy()
            disp_rgb = rgb.detach().cpu().numpy().transpose(1,2,0)
            plt.figure(figsize=(10, 10))
            plt.imshow(disp_rgb)
            plt.scatter(disp_pc_lidar_np[:, 0], disp_pc_lidar_np[:, 1], c=disp_pc_lidar_np[:, 2], s=2)
            plt.title("input display", fontsize=22)
            plt.axis('off')
            plt.show()

            # # gt display
            # disp_gt = depth_gt.detach().cpu().numpy().transpose(1,2,0)
            # plt.figure(figsize=(10, 10))
            # plt.imshow(disp_gt)
            # plt.title("gt display", fontsize=22)
            # plt.axis('off')
            # plt.show()

            # disp_gt1 = dense_depth_img.unsqueeze(0).detach().cpu().numpy().transpose(1,2,0)
            # plt.figure(figsize=(10, 10))
            # plt.imshow(disp_gt1)
            # plt.title("gt display", fontsize=22)
            # plt.axis('off')
            # plt.show()

            disp_gt2 = dense_depth_img_color_gt.detach().cpu().numpy().transpose(1,2,0)
            plt.figure(figsize=(10, 10))
            plt.imshow(disp_gt2, cmap='magma')
            plt.title("gt display", fontsize=22 )
            plt.axis('off')
            plt.show()
            
            if _config['save_image']:
                # save the Lidar pointcloud
                #pcl_lidar = o3.PointCloud()
                pcl_lidar = o3.geometry.PointCloud()
                pc_lidar = pc_lidar.detach().cpu().numpy()
                #pcl_lidar.points = o3.Vector3dVector(pc_lidar.T[:, :3])
                pcl_lidar.points = o3.utility.Vector3dVector(pc_lidar.T[:, :3])

                # o3.draw_geometries(downpcd)
                o3.io.write_point_cloud(pc_lidar_path + '/{}.pcd'.format(batch_idx), pcl_lidar)


            R = quat2mat(sample['rot_error'][idx])
            T = tvector2mat(sample['tr_error'][idx])
            RT_inv = torch.mm(T, R)
            RT = RT_inv.clone().inverse()

            if _config['dataset_type'] =='kitti': 
                if _config['max_depth'] < 80.:
                    pc_rotated = pc_rotated[:, pc_rotated[0, :] < _config['max_depth']].clone()

                pc_rotated = rotate_back(sample['point_cloud'][idx].to(device), RT_inv)  # Pc` = RT * Pc

                depth_img, uv, z, points_index = lidar_project_depth(pc_rotated , sample['calib'][idx].to(device), real_shape)  # image_shape
                depth_img /= _config['max_depth']


            if _config['dataset_type'] == 'nuscenes':
                # N = pc_transformed_gt.shape[1]
                # ones = torch.ones((1, N)).to(pc_transformed_gt.device)
                # homogenous_pc= torch.cat([pc_transformed_gt, ones], dim=0) 

                pc_rotated = rotate_back(pc_transformed_gt, RT_inv)  # Pc` = RT * Pc

                depth_img, uv , z , points_index  = lidar_project_depth_nuscenes(nusc, pc_rotated, cam_data, real_shape)
                depth_img /= _config['max_depth']
            
            lidarOnImage = torch.cat((uv, z), dim=1)
            dense_depth_img = dense_map(lidarOnImage.T , real_shape[1], real_shape[0] , _config['dense_resoltuion']) # argument = (lidarOnImage.T , 1241, 376 , 8)
            dense_depth_img = dense_depth_img.to(dtype=torch.uint8)
            dense_depth_img_color = colormap(dense_depth_img)

            disp_gt3 = dense_depth_img_color.detach().cpu().numpy().transpose(1,2,0)
            plt.figure(figsize=(10, 10))
            plt.imshow(disp_gt3, cmap='magma')
            plt.title("miscalibrated display", fontsize=22 )
            plt.axis('off')
            plt.show()

            # PAD ONLY ON RIGHT AND BOTTOM SIDE
            shape_pad = [0, 0, 0, 0]
            shape_pad[3] = (img_shape[0] - rgb.shape[1])  # // 2
            shape_pad[1] = (img_shape[1] - rgb.shape[2])  # // 2 + 1

            rgb = F.pad(rgb, shape_pad)
            dense_depth_img_color = F.pad(dense_depth_img_color, shape_pad)
            dense_depth_img_color_gt = F.pad(dense_depth_img_color_gt, shape_pad)

            corrs_with_z = corr_gen_withZ (gt_points_index, points_index , gt_uv, uv , gt_z, z, real_shape, rgb_resize_shape, _config["num_kp"] )
            # corrs = torch.cat([corrs_with_z[:, :2], corrs_with_z[:, 3:5]], dim=1)
            corrs = corrs_with_z

            # if _config['outlier_filter'] and sample['depth_img'][0].shape[0] <= _config['outlier_filter_th']:
            #     outlier_filter = True
            # else:
            #     outlier_filter = False
            
            if _config['save_image']:
                # save the RGB input pointcloud
                img = cv2.imread(sample['img_path'][0])
#                 uv_input = sample['depth_img_uv'][0]
#                 pc_input_valid = sample['depth_img_pc_valid'][0]
                R = img[uv[:, 1], uv[:, 0], 0] / 255
                G = img[uv[:, 1], uv[:, 0], 1] / 255
                B = img[uv[:, 1], uv[:, 0], 2] / 255
                pcl_input = o3.geometry.PointCloud()
                pcl_input.points = o3.utility.Vector3dVector(points_index[:, :3])
                pcl_input.colors = o3.utility.Vector3dVector(np.vstack((R, G, B)).T)

                # o3.draw_geometries(downpcd)
                o3.io.write_point_cloud(pc_input_path + '/{}.pcd'.format(batch_idx), pcl_input)

            # batch stack 
            rgb_input.append(rgb)
            lidar_input.append(dense_depth_img_color)
            lidar_input_gt.append(dense_depth_img_color_gt)
            corrs_input.append(corrs)
            pc_rotated_input.append(pc_rotated)
            shape_pad_input.append(shape_pad)
            RTs.append(RT)

        if outlier_filter:
            continue

        rgb_input = torch.stack(rgb_input)
        lidar_input = torch.stack(lidar_input)
        lidar_input_gt = torch.stack(lidar_input_gt)
        rgb_input = F.interpolate(rgb_input, size=[192, 640], mode="bilinear") # lidar 2d depth map input [192,640,1]
        lidar_input = F.interpolate(lidar_input, size=[192, 640], mode="bilinear") # camera input = [192,640,3]
        lidar_input_gt = F.interpolate(lidar_input_gt, size=[192, 640], mode="bilinear") # camera input = [192,640,3]
        corrs_input = torch.stack(corrs_input)

        resized_img = maintain_aspect_ratio_resize(rgb_input, [192, 640])
        resized_lidar_input = maintain_aspect_ratio_resize(lidar_input, [192, 640])
        resized_lidar_input_gt = maintain_aspect_ratio_resize(lidar_input_gt, [192, 640])
        
        corrs_input[:,:,0] = corrs_input[:,:,0]/2    # recaling points for sbs image resizing
        corrs_input[:,:,1] = corrs_input[:,:,1]/2 
        corrs_input[:,:,3] = corrs_input[:,:,3]/2 + 0.5 # recaling points for sbs image resizing
        corrs_input[:,:,4] = corrs_input[:,:,4]/2 

        # queries     = corrs_input[:, :, :2]
        # corr_target = corrs_input[:, :, 2:]
        queries     = corrs_input[:, :, :3]
        corr_target = corrs_input[:, :, 3:]
        
        ####### display input signal #########        
        plt.figure(figsize=(10, 10))
        plt.subplot(311)
        plt.imshow(torchvision.utils.make_grid(rgb_input).permute(1,2,0).cpu().numpy())
        plt.title("camera_input", fontsize=22)
        plt.axis('off')

        plt.subplot(312)
        plt.imshow(torchvision.utils.make_grid(lidar_input_gt).permute(1,2,0).cpu().numpy() , cmap='magma')
        plt.title("calibrated_lidar_input", fontsize=22)
        plt.axis('off') 
 
        plt.subplot(313)
        plt.imshow(torchvision.utils.make_grid(lidar_input).permute(1,2,0).cpu().numpy() , cmap='magma')
        plt.title("mis-calibrated_lidar_input", fontsize=22)
        plt.axis('off')   
          
        # ############ end of display input signal ###################

        #         ####### display input signal #########        
        # plt.figure(figsize=(10, 10))
        # plt.subplot(311)
        # plt.imshow(torchvision.utils.make_grid(resized_img).permute(1,2,0).cpu().numpy())
        # plt.title("camera_input__", fontsize=22)
        # plt.axis('off')

        # plt.subplot(312)
        # plt.imshow(torchvision.utils.make_grid(resized_lidar_input_gt).permute(1,2,0).cpu().numpy() , cmap='magma')
        # plt.title("calibrated_lidar_input__", fontsize=22)
        # plt.axis('off') 
 
        # plt.subplot(313)
        # plt.imshow(torchvision.utils.make_grid(resized_lidar_input).permute(1,2,0).cpu().numpy() , cmap='magma')
        # plt.title("mis-calibrated_lidar_input__", fontsize=22)
        # plt.axis('off')   
          
        # ############# end of display input signal ###################


#         print ('------lidar_input shape---------', lidar_input.shape)
#         print ('------rgb_input shape---------', rgb_input.shape)

        if _config['save_image']:
            out0 = overlay_imgs(rgb_input[0], lidar_input)
#             print ('------out0 shape---------', np.array(out0).shape)
#             out0 = out0[:376, :1241, :]
            out0 = out0[:256, :256, :]
            cv2.imwrite(os.path.join(input_path, sample['rgb_name'][0]), out0[:, :, [2, 1, 0]]*255)
            out1 = overlay_imgs(rgb_input[0], depth_gt[0].unsqueeze(0))
#             out1 = out1[:376, :1241, :]
            out1 = out1[:256, :256, :]
            cv2.imwrite(os.path.join(gt_path, sample['rgb_name'][0]), out1[:, :, [2, 1, 0]]*255)

            depth_img = depth_img.detach().cpu().numpy()
            depth_img = (depth_img / np.max(depth_img)) * 255
            cv2.imwrite(os.path.join(depth_path, sample['rgb_name'][0]), depth_img[0, :376, :1241])

        if show:
            out0 = overlay_imgs(rgb_input[0], lidar_input)
            out1 = overlay_imgs(rgb_input[0], depth_gt[0].unsqueeze(0))
            cv2.imshow("INPUT", out0[:, :, [2, 1, 0]])
            cv2.imshow("GT", out1[:, :, [2, 1, 0]])
            cv2.waitKey(1)
        
        
        rgb = rgb_input.to(device)
        lidar = lidar_input.to(device)
        query = queries.to(device)       
        corr_target_gpu = corr_target.to(device)
        
        target_transl = sample['tr_error'].cuda()
        target_rot = sample['rot_error'].cuda()
        
#         print ('---------predict input rgb shape ----------' , rgb.shape)
#         print ('---------predict input lidar shape --------' , lidar.shape)
        
        # the initial calibration errors before sensor calibration
        RT1 = RTs[0]
        mis_calib = torch.stack(sample['initial_RT'])[1:]
        mis_calib_list.append(mis_calib)

        T_composed = RT1[:3, 3]
        R_composed = quaternion_from_matrix(RT1)
        errors_t[0].append(T_composed.norm().item())
        errors_t2[0].append(T_composed)
        errors_r[0].append(quaternion_distance(R_composed.unsqueeze(0),
                                               torch.tensor([1., 0., 0., 0.], device=R_composed.device).unsqueeze(0),
                                               R_composed.device))
        # rpy_error = quaternion_to_tait_bryan(R_composed)
        rpy_error = mat2xyzrpy(RT1)[3:]

        rpy_error *= (180.0 / 3.141592)
        errors_rpy[0].append(rpy_error)
        log_string += [str(errors_t[0][-1]), str(errors_r[0][-1]), str(errors_t2[0][-1][0].item()),
                       str(errors_t2[0][-1][1].item()), str(errors_t2[0][-1][2].item()),
                       str(errors_rpy[0][-1][0].item()), str(errors_rpy[0][-1][1].item()),
                       str(errors_rpy[0][-1][2].item())]

        # if batch_idx == 0.:
        #     print(f'Initial T_erorr: {errors_t[0]}')
        #     print(f'Initial R_erorr: {errors_r[0]}')
        start = 0
        t1 = time.time()

        # Run model
        with torch.no_grad():
            for iteration in range(start, len(weights)):
                # Run the i-th network
                t1 = time.time()
                if _config['iterative_method'] == 'single_range' or _config['iterative_method'] == 'single':
                    T_predicted, R_predicted , corr_pred , cycle , mask = models[0](rgb, lidar ,query , corr_target_gpu)
                elif _config['iterative_method'] == 'multi_range':
                    T_predicted, R_predicted , corr_pred , cycle , mask = models[iteration](rgb, lidar ,query , corr_target_gpu)
                    print ("enter model inference")
                run_time = time.time() - t1

                if _config['rot_transl_separated'] and iteration == 0:
                    T_predicted = torch.tensor([[0., 0., 0.]], device='cuda')
                if _config['rot_transl_separated'] and iteration == 1:
                    R_predicted = torch.tensor([[1., 0., 0., 0.]], device='cuda')

                # Project the points in the new pose predicted by the i-th network
                R_predicted = quat2mat(R_predicted[0])
                T_predicted = tvector2mat(T_predicted[0])
                RT_predicted = torch.mm(T_predicted, R_predicted)
                RTs.append(torch.mm(RTs[iteration], RT_predicted)) # inv(H_gt)*H_pred_1*H_pred_2*.....H_pred_n
                if iteration == 0:
                    rotated_point_cloud = pc_rotated_input[0]
                else:
                    rotated_point_cloud = rotated_point_cloud

                rotated_point_cloud = rotate_forward(rotated_point_cloud, RT_predicted) # H_pred*X_init

                # depth_img_pred, uv_pred, z_pred, pc_pred_valid = lidar_project_depth(rotated_point_cloud, sample['calib'][0], real_shape , sample['pc_mask'][0], _config['dataset_type']) # image_shape
                # depth_img_pred /= _config['max_depth']
                # depth_pred = F.pad(depth_img_pred, shape_pad_input[0])
                # lidar = depth_pred.unsqueeze(0)
                # lidar_resize = F.interpolate(lidar, size=[256, 512], mode="bilinear")

                if iteration == len(weights)-1 and _config['save_image']:
                    # save the RGB pointcloud
                    img = cv2.imread(sample['img_path'][0])
                    R = img[uv_pred[:, 1], uv_pred[:, 0], 0] / 255
                    G = img[uv_pred[:, 1], uv_pred[:, 0], 1] / 255
                    B = img[uv_pred[:, 1], uv_pred[:, 0], 2] / 255
                    pcl_pred = o3.geometry.PointCloud()
                    pcl_pred.points = o3.utility.Vector3dVector(pc_pred_valid[:, :3])
                    pcl_pred.colors = o3.utility.Vector3dVector(np.vstack((R, G, B)).T)

                    # o3.draw_geometries(downpcd)
                    o3.io.write_point_cloud(pc_pred_path + '/{}.pcd'.format(batch_idx), pcl_pred)


                if _config['save_image']:
                    out2 = overlay_imgs(rgb_input[0], lidar)
                    out2 = out2[:376, :1241, :]
                    cv2.imwrite(os.path.join(os.path.join(pred_path, 'iteration_'+str(iteration+1)),
                                             sample['rgb_name'][0]), out2[:, :, [2, 1, 0]]*255)
                if show:
                    out2 = overlay_imgs(rgb_input[0], lidar)
                    cv2.imshow(f'Pred_Iter_{iteration}', out2[:, :, [2, 1, 0]])
                    cv2.waitKey(1)

                # inv(H_init)*H_pred
                T_composed = RTs[iteration + 1][:3, 3]
                R_composed = quaternion_from_matrix(RTs[iteration + 1])
                errors_t[iteration + 1].append(T_composed.norm().item())
                errors_t2[iteration + 1].append(T_composed)
                errors_r[iteration + 1].append(quaternion_distance(R_composed.unsqueeze(0),
                                                                   torch.tensor([1., 0., 0., 0.], device=R_composed.device).unsqueeze(0),
                                                                   R_composed.device))

                # rpy_error = quaternion_to_tait_bryan(R_composed)
                rpy_error = mat2xyzrpy(RTs[iteration + 1])[3:]
                rpy_error *= (180.0 / 3.141592)
                errors_rpy[iteration + 1].append(rpy_error)
                log_string += [str(errors_t[iteration + 1][-1]), str(errors_r[iteration + 1][-1]),
                               str(errors_t2[iteration + 1][-1][0].item()), str(errors_t2[iteration + 1][-1][1].item()),
                               str(errors_t2[iteration + 1][-1][2].item()), str(errors_rpy[iteration + 1][-1][0].item()),
                               str(errors_rpy[iteration + 1][-1][1].item()), str(errors_rpy[iteration + 1][-1][2].item())]

        run_time = time.time() - t1
        total_time += run_time

        # final calibration error
        all_RTs.append(RTs[-1])
        prev_RT = RTs[-1].inverse()
        prev_tr_error = prev_RT[:3, 3].unsqueeze(0)
        prev_rot_error = quaternion_from_matrix(prev_RT).unsqueeze(0)

        if _config['save_log']:
            log_file.writerow(log_string)

    # Yaw（偏航）：欧拉角向量的y轴
    # Pitch（俯仰）：欧拉角向量的x轴
    # Roll（翻滚）： 欧拉角向量的z轴
    # mis_calib_input[transl_x, transl_y, transl_z, rotx, roty, rotz] Nx6
    mis_calib_input = torch.stack(mis_calib_list)[:, :, 0]
    
    if _config['save_log']:
        log_file.close()
    print("Iterative refinement: ")
    for i in range(len(weights) + 1):
        errors_r[i] = torch.tensor(errors_r[i]).abs() * (180.0 / 3.141592)
        errors_t[i] = torch.tensor(errors_t[i]).abs() * 100

        for k in range(len(errors_rpy[i])):
            # errors_rpy[i][k] = torch.tensor(errors_rpy[i][k])
            # errors_t2[i][k] = torch.tensor(errors_t2[i][k]) * 100
            errors_rpy[i][k] = errors_rpy[i][k].clone().detach().abs()
            errors_t2[i][k] = errors_t2[i][k].clone().detach().abs() * 100

            mean_trans = (errors_t2[i][0].mean() +  errors_t2[i][1].mean() + errors_t2[i][2].mean())/3
            mean_rot = (errors_rpy[i][0].mean() +  errors_rpy[i][1].mean() + errors_rpy[i][2].mean())/3

        # print(f"Iteration {i}: \tMean Translation Error: {errors_t[i].mean():.4f} cm "
        #       f"     Mean Rotation Error: {errors_r[i].mean():.4f} °")
        print(f"Iteration {i}: \tMean Translation Error: {mean_trans:.4f} cm "
              f"     Mean Rotation Error: {mean_rot:.4f} °")
        print(f"Iteration {i}: \tMedian Translation Error: {errors_t[i].median():.4f} cm "
              f"     Median Rotation Error: {errors_r[i].median():.4f} °")
        print(f"Iteration {i}: \tStd. Translation Error: {errors_t[i].std():.4f} cm "
              f"     Std. Rotation Error: {errors_r[i].std():.4f} °\n")

        # translation xyz
        print(f"Iteration {i}: \tMean Translation X Error: {errors_t2[i][0].mean():.4f} cm "
              f"     Median Translation X Error: {errors_t2[i][0].median():.4f} cm "
              f"     Std. Translation X Error: {errors_t2[i][0].std():.4f} cm ")
        print(f"Iteration {i}: \tMean Translation Y Error: {errors_t2[i][1].mean():.4f} cm "
              f"     Median Translation Y Error: {errors_t2[i][1].median():.4f} cm "
              f"     Std. Translation Y Error: {errors_t2[i][1].std():.4f} cm ")
        print(f"Iteration {i}: \tMean Translation Z Error: {errors_t2[i][2].mean():.4f} cm "
              f"     Median Translation Z Error: {errors_t2[i][2].median():.4f} cm "
              f"     Std. Translation Z Error: {errors_t2[i][2].std():.4f} cm \n")

        # rotation rpy[roll pitch yaw]
        print(f"Iteration {i}: \tMean Rotation Roll Error: {errors_rpy[i][0].mean(): .4f} °"
              f"     Median Rotation Roll Error: {errors_rpy[i][0].median():.4f} °"
              f"     Std. Rotation Roll Error: {errors_rpy[i][0].std():.4f} °")
        print(f"Iteration {i}: \tMean Rotation Pitch Error: {errors_rpy[i][1].mean(): .4f} °"
              f"     Median Rotation Pitch Error: {errors_rpy[i][1].median():.4f} °"
              f"     Std. Rotation Pitch Error: {errors_rpy[i][1].std():.4f} °")
        print(f"Iteration {i}: \tMean Rotation Yaw Error: {errors_rpy[i][2].mean(): .4f} °"
              f"     Median Rotation Yaw Error: {errors_rpy[i][2].median():.4f} °"
              f"     Std. Rotation Yaw Error: {errors_rpy[i][2].std():.4f} °\n")


        with open(os.path.join(_config['output'], 'results.txt'),
                  'a', encoding='utf-8') as f:
            f.write(f"Iteration {i}: \n")
            f.write("Translation Error && Rotation Error:\n")
            f.write(f"Iteration {i}: \tMean Translation Error: {errors_t[i].mean():.4f} cm "
                    f"     Mean Rotation Error: {errors_r[i].mean():.4f} °\n")
            f.write(f"Iteration {i}: \tMedian Translation Error: {errors_t[i].median():.4f} cm "
                    f"     Median Rotation Error: {errors_r[i].median():.4f} °\n")
            f.write(f"Iteration {i}: \tStd. Translation Error: {errors_t[i].std():.4f} cm "
                    f"     Std. Rotation Error: {errors_r[i].std():.4f} °\n\n")

            # translation xyz
            f.write("Translation Error XYZ:\n")
            f.write(f"Iteration {i}: \tMean Translation X Error: {errors_t2[i][0].mean():.4f} cm "
                    f"     Median Translation X Error: {errors_t2[i][0].median():.4f} cm "
                    f"     Std. Translation X Error: {errors_t2[i][0].std():.4f} cm \n")
            f.write(f"Iteration {i}: \tMean Translation Y Error: {errors_t2[i][1].mean():.4f} cm "
                    f"     Median Translation Y Error: {errors_t2[i][1].median():.4f} cm "
                    f"     Std. Translation Y Error: {errors_t2[i][1].std():.4f} cm \n")
            f.write(f"Iteration {i}: \tMean Translation Z Error: {errors_t2[i][2].mean():.4f} cm "
                    f"     Median Translation Z Error: {errors_t2[i][2].median():.4f} cm "
                    f"     Std. Translation Z Error: {errors_t2[i][2].std():.4f} cm \n\n")

            # rotation rpy[roll pitch yaw]
            f.write("Rotation Error RPY:\n")
            f.write(f"Iteration {i}: \tMean Rotation Roll Error: {errors_rpy[i][0].mean(): .4f} °"
                    f"     Median Rotation Roll Error: {errors_rpy[i][0].median():.4f} °"
                    f"     Std. Rotation Roll Error: {errors_rpy[i][0].std():.4f} °\n")
            f.write(f"Iteration {i}: \tMean Rotation Pitch Error: {errors_rpy[i][1].mean(): .4f} °"
                    f"     Median Rotation Pitch Error: {errors_rpy[i][1].median():.4f} °"
                    f"     Std. Rotation Pitch Error: {errors_rpy[i][1].std():.4f} °\n")
            f.write(f"Iteration {i}: \tMean Rotation Yaw Error: {errors_rpy[i][2].mean(): .4f} °"
                    f"     Median Rotation Yaw Error: {errors_rpy[i][2].median():.4f} °"
                    f"     Std. Rotation Yaw Error: {errors_rpy[i][2].std():.4f} °\n\n\n")

    for i in range(len(errors_t2)):
        errors_t2[i] = torch.stack(errors_t2[i]).abs() / 100
        errors_rpy[i] = torch.stack(errors_rpy[i]).abs()

        
    plot_x = np.zeros((mis_calib_input.shape[0], 2))
    plot_x[:, 0] = mis_calib_input[:, 0].cpu().numpy()
    plot_x[:, 1] = errors_t2[-1][:, 0].cpu().numpy()
    plot_x = plot_x[np.lexsort(plot_x[:, ::-1].T)]

    plot_y = np.zeros((mis_calib_input.shape[0], 2))
    plot_y[:, 0] = mis_calib_input[:, 1].cpu().numpy()
    plot_y[:, 1] = errors_t2[-1][:, 1].cpu().numpy()
    plot_y = plot_y[np.lexsort(plot_y[:, ::-1].T)]

    plot_z = np.zeros((mis_calib_input.shape[0], 2))
    plot_z[:, 0] = mis_calib_input[:, 2].cpu().numpy()
    plot_z[:, 1] = errors_t2[-1][:, 2].cpu().numpy()
    plot_z = plot_z[np.lexsort(plot_z[:, ::-1].T)]

    N_interval = plot_x.shape[0] // N
    plot_x = plot_x[::N_interval]
    plot_y = plot_y[::N_interval]
    plot_z = plot_z[::N_interval]

    plt.plot(plot_x[:, 0], plot_x[:, 1], c='red', label='X')
    plt.plot(plot_y[:, 0], plot_y[:, 1], c='blue', label='Y')
    plt.plot(plot_z[:, 0], plot_z[:, 1], c='green', label='Z')
    # plt.legend(loc='best')

    if _config['out_fig_lg'] == 'EN':
        plt.xlabel('Miscalibration (m)', font_EN)
        plt.ylabel('Absolute Error (m)', font_EN)
        plt.legend(loc='best', prop=font_EN)
    elif _config['out_fig_lg'] == 'CN':
        plt.xlabel('初始标定外参偏差/米', font_CN)
        plt.ylabel('绝对误差/米', font_CN)
        plt.legend(loc='best', prop=font_CN)

    plt.xticks(fontproperties='Times New Roman', size=plt_size)
    plt.yticks(fontproperties='Times New Roman', size=plt_size)

    plt.savefig(os.path.join(results_path, 'xyz_plot.png'))
    plt.close('all')
    # plt.axis('off')
    # plt.show() 

    errors_t = errors_t[-1].numpy()
    errors_t = np.sort(errors_t, axis=0)[:-10] # 去掉一些异常值
    # plt.title('Calibration Translation Error Distribution')
    plt.hist(errors_t / 100, bins=50)
    # ax = plt.gca()
    # ax.set_xlabel('Absolute Translation Error (m)')
    # ax.set_ylabel('Number of instances')
    # ax.set_xticks([0.00, 0.25, 0.00, 0.25, 0.50])

    if _config['out_fig_lg'] == 'EN':
        plt.xlabel('Absolute Translation Error (m)', font_EN)
        plt.ylabel('Number of instances', font_EN)
    elif _config['out_fig_lg'] == 'CN':
        plt.xlabel('绝对平移误差/米', font_CN)
        plt.ylabel('实验序列数目/个', font_CN)
    plt.xticks(fontproperties='Times New Roman', size=plt_size)
    plt.yticks(fontproperties='Times New Roman', size=plt_size)

    plt.savefig(os.path.join(results_path, 'translation_error_distribution.png'))
    plt.close('all')

    # rotation error
    # fig = plt.figure(figsize=(6, 3))  # 이미지 크기 설정 figsize=(6,3)
    # plt.title('Calibration Rotation Error')
    plot_pitch = np.zeros((mis_calib_input.shape[0], 2))
    plot_pitch[:, 0] = mis_calib_input[:, 3].cpu().numpy() * (180.0 / 3.141592)
    plot_pitch[:, 1] = errors_rpy[-1][:, 1].cpu().numpy()
    plot_pitch = plot_pitch[np.lexsort(plot_pitch[:, ::-1].T)]

    plot_yaw = np.zeros((mis_calib_input.shape[0], 2))
    plot_yaw[:, 0] = mis_calib_input[:, 4].cpu().numpy() * (180.0 / 3.141592)
    plot_yaw[:, 1] = errors_rpy[-1][:, 2].cpu().numpy()
    plot_yaw = plot_yaw[np.lexsort(plot_yaw[:, ::-1].T)]

    plot_roll = np.zeros((mis_calib_input.shape[0], 2))
    plot_roll[:, 0] = mis_calib_input[:, 5].cpu().numpy() * (180.0 / 3.141592)
    plot_roll[:, 1] = errors_rpy[-1][:, 0].cpu().numpy()
    plot_roll = plot_roll[np.lexsort(plot_roll[:, ::-1].T)]

    N_interval = plot_roll.shape[0] // N
    plot_pitch = plot_pitch[::N_interval]
    plot_yaw = plot_yaw[::N_interval]
    plot_roll = plot_roll[::N_interval]

    # Yaw : 오일러 각도 벡터의 y 축
    # Pitch : 오일러 각도 벡터의 x 축
    # Roll : 오일러 각도 벡터의 z 축

    if _config['out_fig_lg'] == 'EN':
        plt.plot(plot_yaw[:, 0], plot_yaw[:, 1], c='red', label='Yaw(Y)')
        plt.plot(plot_pitch[:, 0], plot_pitch[:, 1], c='blue', label='Pitch(X)')
        plt.plot(plot_roll[:, 0], plot_roll[:, 1], c='green', label='Roll(Z)')
        plt.xlabel('Miscalibration (°)', font_EN)
        plt.ylabel('Absolute Error (°)', font_EN)
        plt.legend(loc='best', prop=font_EN)
    elif _config['out_fig_lg'] == 'CN':
        plt.plot(plot_yaw[:, 0], plot_yaw[:, 1], c='red', label='偏航角')
        plt.plot(plot_pitch[:, 0], plot_pitch[:, 1], c='blue', label='俯仰角')
        plt.plot(plot_roll[:, 0], plot_roll[:, 1], c='green', label='翻滚角')
        plt.xlabel('初始标定外参偏差/度', font_CN)
        plt.ylabel('绝对误差/度', font_CN)
        plt.legend(loc='best', prop=font_CN)

    plt.xticks(fontproperties='Times New Roman', size=plt_size)
    plt.yticks(fontproperties='Times New Roman', size=plt_size)
    plt.savefig(os.path.join(results_path, 'rpy_plot.png'))
    plt.close('all')

    errors_r = errors_r[-1].numpy()
    errors_r = np.sort(errors_r, axis=0)[:-10] # 去掉一些异常值
    # np.savetxt('rot_error.txt', arr_, fmt='%0.8f')
    # print('max rotation_error: {}'.format(max(errors_r)))
    # plt.title('Calibration Rotation Error Distribution')
    plt.hist(errors_r, bins=50)
    #plt.xlim([0, 1.5])  # x轴边界
    #plt.xticks([0.0, 0.3, 0.6, 0.9, 1.2, 1.5])  # 设置x刻度
    # ax = plt.gca()

    if _config['out_fig_lg'] == 'EN':
        plt.xlabel('Absolute Rotation Error (°)', font_EN)
        plt.ylabel('Number of instances', font_EN)
    elif _config['out_fig_lg'] == 'CN':
        plt.xlabel('绝对旋转误差/度', font_CN)
        plt.ylabel('实验序列数目/个', font_CN)
    plt.xticks(fontproperties='Times New Roman', size=plt_size)
    plt.yticks(fontproperties='Times New Roman', size=plt_size)
    plt.savefig(os.path.join(results_path, 'rotation_error_distribution.png'))
    plt.close('all')


    if _config["save_name"] is not None:
        torch.save(torch.stack(errors_t).cpu().numpy(), f'./results_for_paper/{_config["save_name"]}_errors_t')
        torch.save(torch.stack(errors_r).cpu().numpy(), f'./results_for_paper/{_config["save_name"]}_errors_r')
        torch.save(torch.stack(errors_t2).cpu().numpy(), f'./results_for_paper/{_config["save_name"]}_errors_t2')
        torch.save(torch.stack(errors_rpy).cpu().numpy(), f'./results_for_paper/{_config["save_name"]}_errors_rpy')

    avg_time = total_time / len(TestImgLoader)
    print("average runing time on {} iteration: {} s".format(len(weights), avg_time))
    print("End")


# In[10]:

ex.run()

