import torch
import numpy as np
import torch.nn.functional as F

import matplotlib as mpl
import matplotlib.cm as cm
import scipy
import skimage
# from pypardiso import spsolve

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
    # pcl_z_no_mask = pcl_z
    pcl_uv = pcl_uv[mask]
    pcl_z = pcl_z[mask]
    pcl_uv = torch.tensor(pcl_uv, dtype=torch.int64)
    # pcl_uv = pcl_uv.astype(np.uint32)
    # pcl_uv_no_mask  = pcl_uv_no_mask.astype(np.uint32) 
    
    pcl_z = pcl_z.reshape(-1, 1)

    depth_img = np.zeros((img_shape[0], img_shape[1], 1))
    depth_img = torch.from_numpy(depth_img.astype(np.float32)).cuda()
    depth_img[pcl_uv[:, 1], pcl_uv[:, 0]] = pcl_z  
    depth_img = depth_img.permute(2, 0, 1)
    points_index = torch.arange(pcl_uv_no_mask.shape[0], device=pcl_uv_no_mask.device)[mask]

    # points_index = np.arange(pcl_uv_no_mask.shape[0])[mask]
    # points_index1 = np.arange(pcl_uv_no_mask.shape[0])[mask1]

    return depth_img, pcl_uv , pcl_z , points_index 
    
def trim_corrs(in_corrs):
    length = in_corrs.shape[0]
#         print ("number of keypoint before trim : {}".format(length))
    if length >= self.num_kp:
        mask = np.random.choice(length, self.num_kp)
        return in_corrs[mask]
    else:
        mask = np.random.choice(length, self.num_kp - length)
        return np.concatenate([in_corrs, in_corrs[mask]], axis=0)

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
    # pairwise_distance = F.pairwise_distance(x2, y2)
    # idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    # top_indices = torch.topk(pairwise_distance.flatten(), k=k, largest=False)
    # top_indices = top_indices.indices
    # indices = np.unravel_index(top_indices, pairwise_distance.shape)
    # top_indices = np.asarray(top_indices).T
    
    #### 가장 먼 포인트 들 뽑기 #########
    idx ,_ = farthest_point_sampling(x2,k)

    top_x = x2[idx]
    top_y = y2[idx]
    # print ("x point of z =" , top_x[3])
    # print ("y point of z =" , top_y[3])
    # top_y[:, 2] =  1- top_y[:, 2] # 세 번째 열의 값에서 1을 빼기 
    # print ("y point of rev z =" , top_y[3])
    
    corrs = torch.cat([top_x,top_y] ,dim=1) 
        
    return idx , corrs

def two_images_side_by_side(img_a, img_b):
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

# From Github https://github.com/balcilar/DenseDepthMap
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
    vmax = np.percentile(disp_np, 95)
    normalizer = mpl.colors.Normalize(vmin=disp_np.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')  #magma, plasma, etc.
    colormapped_im = (mapper.to_rgba(disp_np)[:, :, :3])
    # return colormapped_im.transpose(2, 0, 1)
    colormapped_tensor = torch.from_numpy(colormapped_im).permute(2, 0, 1).to(dtype=torch.float32).cuda()
    # colormapped_tensor = torch.from_numpy(colormapped_im).
    return colormapped_tensor

# corr dataset generation 
def corr_gen( gt_points_index, points_index, gt_uv, uv , num_kp = 500) :
    
    inter_gt_uv_mask = np.in1d(gt_points_index , points_index)
    inter_uv_mask    = np.in1d(points_index , gt_points_index)
    gt_uv = gt_uv[inter_gt_uv_mask]
    uv    = uv[inter_uv_mask] 
    corrs = np.concatenate([gt_uv, uv], axis=1)
    corrs = torch.tensor(corrs)
    
    ## corrs 384*1280 image(original image shape) normalization
    corrs[:, 0] = (0.5*corrs[:, 0])/1280
    corrs[:, 1] = (0.5*corrs[:, 1])/384
    corrs[:, 2] = (0.5*corrs[:, 2])/1280 + 0.5        
    corrs[:, 3] = (0.5*corrs[:, 3])/384   
    
    if corrs.shape[0] <= num_kp :
        corrs = torch.zeros(num_kp, 4)
        corrs[:, 2] = corrs[:, 2] + 0.5

    corrs_knn_idx = knn(corrs[:,:2], corrs[:,2:], num_kp) # knn 2d point-cloud trim
    corrs = corrs[corrs_knn_idx]               

    assert (0.0 <= corrs[:, 0]).all() and (corrs[:, 0] <= 0.5).all()
    assert (0.0 <= corrs[:, 1]).all() and (corrs[:, 1] <= 1.0).all()
    assert (0.5 <= corrs[:, 2]).all() and (corrs[:, 2] <= 1.0).all()
    assert (0.0 <= corrs[:, 3]).all() and (corrs[:, 3] <= 1.0).all()
    
    return corrs

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

# for displying correspondence matching 
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


# fill_depth_colorization.m
# Preprocesses the kinect depth image using a gray scale version of the
# RGB image as a weighting for the smoothing. This code is a slight
# adaptation of Anat Levin's colorization code:
#
# See: www.cs.huji.ac.il/~yweiss/Colorization/
#
# Args:
#  imgRgb - HxWx3 matrix, the rgb image for the current frame. This must
#      be between 0 and 1.
#  imgDepth - HxW matrix, the depth image for the current frame in
#       absolute (meters) space.
#  alpha - a penalty value between 0 and 1 for the current depth values.

def fill_depth_colorization(imgRgb=None, imgDepthInput=None, alpha=1):
	imgIsNoise = imgDepthInput == 0
	maxImgAbsDepth = np.max(imgDepthInput)
	imgDepth = imgDepthInput / maxImgAbsDepth
	imgDepth[imgDepth > 1] = 1
	(H, W) = imgDepth.shape
	numPix = H * W
	indsM = np.arange(numPix).reshape((W, H)).transpose()
	knownValMask = (imgIsNoise == False).astype(int)
	grayImg = skimage.color.rgb2gray(imgRgb)
	winRad = 1
	len_ = 0
	absImgNdx = 0
	len_window = (2 * winRad + 1) ** 2
	len_zeros = numPix * len_window

	cols = np.zeros(len_zeros) - 1
	rows = np.zeros(len_zeros) - 1
	vals = np.zeros(len_zeros) - 1
	gvals = np.zeros(len_window) - 1

	for j in range(W):
		for i in range(H):
			nWin = 0
			for ii in range(max(0, i - winRad), min(i + winRad + 1, H)):
				for jj in range(max(0, j - winRad), min(j + winRad + 1, W)):
					if ii == i and jj == j:
						continue

					rows[len_] = absImgNdx
					cols[len_] = indsM[ii, jj]
					gvals[nWin] = grayImg[ii, jj]

					len_ = len_ + 1
					nWin = nWin + 1

			curVal = grayImg[i, j]
			gvals[nWin] = curVal
			c_var = np.mean((gvals[:nWin + 1] - np.mean(gvals[:nWin+ 1])) ** 2)

			csig = c_var * 0.6
			mgv = np.min((gvals[:nWin] - curVal) ** 2)
			if csig < -mgv / np.log(0.01):
				csig = -mgv / np.log(0.01)

			if csig < 2e-06:
				csig = 2e-06

			gvals[:nWin] = np.exp(-(gvals[:nWin] - curVal) ** 2 / csig)
			gvals[:nWin] = gvals[:nWin] / sum(gvals[:nWin])
			vals[len_ - nWin:len_] = -gvals[:nWin]

	  		# Now the self-reference (along the diagonal).
			rows[len_] = absImgNdx
			cols[len_] = absImgNdx
			vals[len_] = 1  # sum(gvals(1:nWin))

			len_ = len_ + 1
			absImgNdx = absImgNdx + 1

	vals = vals[:len_]
	cols = cols[:len_]
	rows = rows[:len_]
	A = scipy.sparse.csr_matrix((vals, (rows, cols)), (numPix, numPix))

	rows = np.arange(0, numPix)
	cols = np.arange(0, numPix)
	vals = (knownValMask * alpha).transpose().reshape(numPix)
	G = scipy.sparse.csr_matrix((vals, (rows, cols)), (numPix, numPix))

	A = A + G
	b = np.multiply(vals.reshape(numPix), imgDepth.flatten('F'))

	#print ('Solving system..')

	new_vals = spsolve(A, b)
	new_vals = np.reshape(new_vals, (H, W), 'F')

	#print ('Done.')

	denoisedDepthImg = new_vals * maxImgAbsDepth
    
	output = denoisedDepthImg.reshape((H, W)).astype('float32')

	output = np.multiply(output, (1-knownValMask)) + imgDepthInput
    
	return output 