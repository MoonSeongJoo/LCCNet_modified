#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -------------------------------------------------------------------
# Copyright (C) 2020 Università degli studi di Milano-Bicocca, iralab
# Author: Daniele Cattaneo (d.cattaneo10@campus.unimib.it)
# Released under Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# http://creativecommons.org/licenses/by-nc-sa/4.0/
# -------------------------------------------------------------------
import torch
from torch import nn as nn

from quaternion_distances import quaternion_distance
from utils import quat2mat, rotate_back, rotate_forward, tvector2mat, quaternion_from_matrix
import numpy as np

class GeometricLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sx = torch.nn.Parameter(torch.Tensor([0.0]), requires_grad=True)
        self.sq = torch.nn.Parameter(torch.Tensor([-3.0]), requires_grad=True)
        self.transl_loss = nn.SmoothL1Loss(reduction='none')

    def forward(self, target_transl, target_rot, transl_err, rot_err):
        loss_transl = self.transl_loss(transl_err, target_transl).sum(1).mean()
        loss_rot = quaternion_distance(rot_err, target_rot, rot_err.device).mean()
        total_loss = torch.exp(-self.sx) * loss_transl + self.sx
        total_loss += torch.exp(-self.sq) * loss_rot + self.sq
        return total_loss


class ProposedLoss(nn.Module):
    def __init__(self, rescale_trans, rescale_rot):
        super(ProposedLoss, self).__init__()
        self.rescale_trans = rescale_trans
        self.rescale_rot = rescale_rot
        self.transl_loss = nn.SmoothL1Loss(reduction='none')

    def forward(self, target_transl, target_rot, transl_err, rot_err):
        loss_transl = 0.
        if self.rescale_trans != 0.:
            loss_transl = self.transl_loss(transl_err, target_transl).sum(1).mean()
        loss_rot = 0.
        if self.rescale_rot != 0.:
            loss_rot = quaternion_distance(rot_err, target_rot, rot_err.device).mean()
        total_loss = self.rescale_trans*loss_transl + self.rescale_rot*loss_rot
        return total_loss


class L1Loss(nn.Module):
    def __init__(self, rescale_trans, rescale_rot):
        super(L1Loss, self).__init__()
        self.rescale_trans = rescale_trans
        self.rescale_rot = rescale_rot
        self.transl_loss = nn.SmoothL1Loss(reduction='none')

    def forward(self, target_transl, target_rot, transl_err, rot_err):
        loss_transl = self.transl_loss(transl_err, target_transl).sum(1).mean()
        loss_rot = self.transl_loss(rot_err, target_rot).sum(1).mean()
        total_loss = self.rescale_trans*loss_transl + self.rescale_rot*loss_rot
        return total_loss


class DistancePoints3D(nn.Module):
    def __init__(self):
        super(DistancePoints3D, self).__init__()

    def forward(self, point_clouds, target_transl, target_rot, transl_err, rot_err):
        """
        Points Distance Error
        Args:
            point_cloud: list of B Point Clouds, each in the relative GT frame
            transl_err: network estimate of the translations
            rot_err: network estimate of the rotations
        Returns:
            The mean distance between 3D points
        """
        #start = time.time()
        total_loss = torch.tensor([0.0]).to(transl_err.device)
        for i in range(len(point_clouds)):
            point_cloud_gt = point_clouds[i].to(transl_err.device)
            point_cloud_out = point_clouds[i].clone()

            R_target = quat2mat(target_rot[i])
            T_target = tvector2mat(target_transl[i])
            RT_target = torch.mm(T_target, R_target)

            R_predicted = quat2mat(rot_err[i])
            T_predicted = tvector2mat(transl_err[i])
            RT_predicted = torch.mm(T_predicted, R_predicted)

            RT_total = torch.mm(RT_target.inverse(), RT_predicted)

            point_cloud_out = rotate_forward(point_cloud_out, RT_total)

            error = (point_cloud_out - point_cloud_gt).norm(dim=0)
            error.clamp(100.)
            total_loss += error.mean()

        #end = time.time()
        #print("3D Distance Time: ", end-start)

        return total_loss/target_transl.shape[0]

class CombinedLoss(nn.Module):
    def __init__(self, rescale_trans, rescale_rot , weight_point_cloud):
        super(CombinedLoss, self).__init__()
        self.rescale_trans = rescale_trans
        self.rescale_rot = rescale_rot
        self.transl_loss = nn.SmoothL1Loss(reduction='none')
        self.rot_loss = nn.SmoothL1Loss(reduction='none')         
        
        self.weight_corr = 0.05 #init 0.05
        self.weight_point_cloud = 0.2 #init 0.2
        self.weight_local_transl = 200.0 # clone init : 10
        self.weight_local_rot = 1.0 # clone init : 10
        self.weight_t_mae = 20.0  #init :3
        self.weight_quaternion = 0.1  #init :0.1
        self.loss = {} 
        
        print ( "------- devide weght rot/trans--------")
        print ( "------- loss weight corr --------" , self.weight_corr )
        print ( "------- loss weight point cloud -------- " , self.weight_point_cloud )
        print ( "------- loss weight quaternion rot --------" , self.weight_quaternion)
        print ( "------- loss weight t_mae --------" , self.weight_t_mae)
        print ( "------- loss weight clone trans --------" , self.weight_local_transl)
        print ( "------- loss weight clone rot --------" , self.weight_local_rot)
        print ( "-------------------------" )

#     def forward(self, point_clouds, target_transl, target_rot, transl_err, rot_err):
        
#         # Regression loss
#         loss_transl = self.transl_loss(transl_err, target_transl).sum(1).mean()
#         loss_rot = self.transl_loss(rot_err, target_rot).sum(1).mean()
#         total_loss_lt = self.rescale_trans*loss_transl + self.rescale_rot*loss_rot
        
#         # Point Cloud Distance Loss
#         total_loss = torch.tensor([0.0]).to(transl_err.device)
#         for i in range(len(point_clouds)):
#             point_cloud_gt = point_clouds[i].to(transl_err.device)
#             point_cloud_out = point_clouds[i].clone()

#             R_target = quat2mat(target_rot[i])
#             T_target = tvector2mat(target_transl[i])
#             RT_target = torch.mm(T_target, R_target)

#             R_predicted = quat2mat(rot_err[i])
#             T_predicted = tvector2mat(transl_err[i])
#             RT_predicted = torch.mm(T_predicted, R_predicted)

#             RT_total = torch.mm(RT_target.inverse(), RT_predicted)
            
#             point_cloud_out = point_cloud_out.cuda()
#             point_cloud_out = rotate_forward(point_cloud_out, RT_total)

#             error = (point_cloud_out - point_cloud_gt).norm(dim=0)
#             error.clamp(100.)
#             total_loss += error.mean()
        
#         total_loss_lp = total_loss/target_transl.shape[0]
        
#         loss = total_loss_lt + self.weight_point_cloud*total_loss_lp
        
#         return loss
    def square_distance(self, pcd1, pcd2):
        """
        Squared distance between any two points in the two point clouds.
        """
        return torch.sum((pcd1[:, :, None, :].detach() - pcd2[:, None, :, :].detach()) ** 2, dim=-1)
        # return torch.sum((pcd1[:, :, :, None] - pcd2[:, :, None, :]) ** 2, dim=-2)

    def forward(self, point_clouds, current_source, exp_transl_seq, exp_rot_seq ,transl_err, rot_err , corr_target , corr_pred , queries , cycle , mask, pos_final, pose_target):
        """
        The Combination of Pose Error and Points Distance Error
        Args:
            point_cloud: list of B Point Clouds, each in the relative GT frame
            target_transl: groundtruth of the translations
            target_rot: groundtruth of the rotations
            transl_err: network estimate of the translations
            rot_err: network estimate of the rotations
        Returns:
            The combination loss of Pose error and the mean distance between 3D points
        """
        # Calculate the quaternion loss between the final pose and the ground truth
        R_composed_target = torch.stack([quaternion_from_matrix(pose_target[i, :]) for i in range(pose_target.shape[0])], dim = 0)
        R_composed = torch.stack([quaternion_from_matrix(pos_final[i, :]) for i in range(pos_final.shape[0])], dim = 0)

        loss_transl = 0.
        if self.rescale_trans != 0.:
            loss_clone_transl = self.transl_loss(transl_err, exp_transl_seq).sum(1).mean()
            loss_clone_rot = self.rot_loss(rot_err, exp_rot_seq).sum(1).mean()
            # clone_loss = loss_transl + loss_simple_rot
        loss_rot = 0.
        if self.rescale_rot != 0.:
            loss_rot = quaternion_distance(R_composed, R_composed_target, R_composed.device)
            loss_rot = loss_rot.abs() * (180.0/np.pi)
        
        # # start = time.time()
        # point_clouds_loss = torch.tensor([0.0]).to(transl_err.device)
        # for i in range(len(point_clouds)):
        #     point_cloud_gt = point_clouds[i].to(transl_err.device)
        #     point_cloud_out = point_clouds[i].clone()

        #     R_target = quat2mat(target_rot[i])
        #     T_target = tvector2mat(target_transl[i])
        #     RT_target = torch.mm(T_target, R_target)

        #     R_predicted = quat2mat(rot_err[i])
        #     T_predicted = tvector2mat(transl_err[i])
        #     RT_predicted = torch.mm(T_predicted, R_predicted)

        #     RT_total = torch.mm(RT_target.inverse(), RT_predicted)
        #     point_cloud_out = point_cloud_out.cuda()
        #     point_cloud_out = rotate_forward(point_cloud_out, RT_total)

        #     error = (point_cloud_out - point_cloud_gt).norm(dim=0)
        #     error.clamp(100.)
        #     point_clouds_loss += error.mean()

        # Calculate the translation loss between the final pose and the target
        t_gt = pose_target[:, :3, 3]
        t_pred = pos_final[:, :3, 3]
        loss_t_mae = torch.abs(t_gt - t_pred).mean(dim=1)

        # point cloud distance loss
        rand_idxs = np.random.choice(current_source.shape[2], 1024, replace=False)
        src_transformed_samp = current_source[:,:, rand_idxs]
        ref_clean_samp = point_clouds[:, :, rand_idxs]
        dist = torch.min(self.square_distance(src_transformed_samp.permute(0,2,1), ref_clean_samp.permute(0,2,1)), dim=-1)[0]
        chamfer_dist = torch.mean(dist, dim=1).reshape(-1, 1)
        geo_loss = chamfer_dist.mean()
        
        corr_loss = torch.nn.functional.mse_loss(corr_pred, corr_target)
        
        if mask.sum() > 0:
#             print('enter cyclic loss sum')
            cycle_loss = torch.nn.functional.mse_loss(cycle[mask], queries[mask])
            corr_loss += cycle_loss    
            corr_loss = corr_loss + cycle_loss
        
        #end = time.time()
        #print("3D Distance Time: ", end-start)
        # total_loss = self.weight_pose * pose_loss +\
        #              self.weight_point_cloud * (point_clouds_loss/target_transl.shape[0]) + self.weight_corr * corr_loss
        # total_loss = self.weight_pose * pose_loss +\
        #              self.weight_point_cloud * (point_clouds_loss/target_transl.shape[0]) 
        # total_loss = self.weight_rot* loss_rot  + self.weight_trans* loss_transl + self.weight_corr * corr_loss +\
        #              self.weight_point_cloud * (point_clouds_loss/target_transl.shape[0]) 
        # total_loss = self.weight_clone* clone_loss + self.weight_rot* loss_rot + self.weight_t_mae* loss_t_mae + self.weight_corr * corr_loss +\
        #              self.weight_point_cloud * geo_loss 
        # total_loss = self.weight_clone* clone_loss + self.weight_rot* loss_rot + self.weight_t_mae* loss_t_mae +self.weight_corr * corr_loss +\
        #             self.weight_point_cloud * (point_clouds_loss/target_transl.shape[0])
        total_loss = self.weight_local_rot* loss_clone_rot+ self.weight_local_transl* loss_clone_transl + self.weight_quaternion * loss_rot + self.weight_t_mae* loss_t_mae +self.weight_corr * corr_loss +\
                    self.weight_point_cloud * geo_loss
        # total_loss = self.weight_clone* clone_loss + self.weight_rot* loss_rot + self.weight_t_mae* loss_t_mae + self.weight_point_cloud * geo_loss # corr freeze
        
        self.loss['total_loss'] = total_loss.sum()/point_clouds.shape[0]
        self.loss['clone_loss_trans'] = loss_clone_transl
        self.loss['clone_loss_rot'] = loss_clone_rot
        self.loss['loss_rot'] = loss_rot.sum()/point_clouds.shape[0]
        self.loss['loss_t_mae'] = loss_t_mae.sum()/point_clouds.shape[0]
        self.loss['corr_loss'] = corr_loss
        self.loss['point_clouds_loss'] = geo_loss
        
        return self.loss