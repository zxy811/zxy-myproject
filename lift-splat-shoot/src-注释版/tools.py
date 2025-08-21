"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import os#拼接路径
import numpy as np#数组矩阵计算
import torch
import torchvision
from tqdm import tqdm#显示进度条可视化
from pyquaternion import Quaternion#计算四元矩阵
from PIL import Image#图像处理计算
from functools import reduce#累计计算
import matplotlib as mpl#导入 Python 的绘图库 matplotlib，并命名为 mpl
mpl.use('Agg')#生成图像文件而不是在图形界面上显示
import matplotlib.pyplot as plt#画图工具
from nuscenes.utils.data_classes import LidarPointCloud#加载、过滤、变换 LiDAR 点云的核心工具
from nuscenes.utils.geometry_utils import transform_matrix#构成四元变换矩阵同时可用于坐标变换
from nuscenes.map_expansion.map_api import NuScenesMap#用于读取高精地图

#聚合多帧雷达点云到参考帧自车坐标系
def get_lidar_data(nusc, sample_rec, nsweeps, min_distance):
    """
    Returns at most nsweeps of lidar in the ego frame.
    Returned tensor is 5(x, y, z, reflectance, dt) x N
    Adapted from https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/data_classes.py#L56
    """
    points = np.zeros((5, 0))#这是一个5行0列的数组，用于存储点云数据

    # Get reference pose and timestamp.
    #获取参考帧的姿态和时间戳
    ref_sd_token = sample_rec['data']['LIDAR_TOP']
    ref_sd_rec = nusc.get('sample_data', ref_sd_token)
    ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
    ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
    ref_time = 1e-6 * ref_sd_rec['timestamp']
    #从全局坐标转移到自车的坐标变换
    # Homogeneous transformation matrix from global to _current_ ego car frame.
    car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
                                        inverse=True)

    # Aggregate current and previous sweeps.
    #多帧聚合循环，从当前帧向前追溯
    sample_data_token = sample_rec['data']['LIDAR_TOP']
    current_sd_rec = nusc.get('sample_data', sample_data_token)
    for _ in range(nsweeps):
        #加载点云并过滤近距离点
        # Load up the pointcloud and remove points close to the sensor.
        current_pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, current_sd_rec['filename']))
        current_pc.remove_close(min_distance)

        # 获取当前帧自车在全局坐标系下面的位姿
        current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
        #将当前帧自车坐标系转换到全局坐标系的一个变换矩阵
        global_from_car = transform_matrix(current_pose_rec['translation'],
                                            Quaternion(current_pose_rec['rotation']), inverse=False)

        # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
        #将雷达传感器转变到当前帧自车坐标系
        current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
        car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                            inverse=False)

        # Fuse four transformation matrices into one and perform transform.
        #car_from_global是参考帧的自车坐标系，因为现在是融合多个帧的点云所以需要参考帧
        trans_matrix = reduce(np.dot, [car_from_global, global_from_car, car_from_current])
        current_pc.transform(trans_matrix)

        # 记录当前帧与参考帧之间的时间差
        time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']
        #将当前每一帧点云带上一个时间戳差的信息
        times = time_lag * np.ones((1, current_pc.nbr_points()))
        #将时间戳差添加到每个点后面
        new_points = np.concatenate((current_pc.points, times), 0)
        #多帧融合点云数据
        points = np.concatenate((points, new_points), 1)

        # 判断当前帧是否还有上一帧
        if current_sd_rec['prev'] == '':
            break
        else:
            current_sd_rec = nusc.get('sample_data', current_sd_rec['prev'])

    return points

#自车到相机
def ego_to_cam(points, rot, trans, intrins):
    """Transform points (3 x N) from ego frame into a pinhole camera
    """
    points = points - trans.unsqueeze(1)#unsqueeze(1)是增加一列
    points = rot.permute(1, 0).matmul(points)#permute(1, 0)是转置

    points = intrins.matmul(points)#matmul是乘
    points[:2] /= points[2:3]

    return points

#相机到自车
def cam_to_ego(points, rot, trans, intrins):
    """Transform points (3 x N) from pinhole camera with depth
    to the ego frame
    """
    points = torch.cat((points[:2] * points[2:3], points[2:3]))
    points = intrins.inverse().matmul(points)

    points = rot.matmul(points)
    points += trans.unsqueeze(1)

    return points

#判断点是否在图像内
def get_only_in_img_mask(pts, H, W):
    """pts should be 3 x N
    """
    return (pts[2] > 0) &\
        (pts[0] > 1) & (pts[0] < W - 1) &\
        (pts[1] > 1) & (pts[1] < H - 1)

#旋转矩阵图像增强时候进行使用，得到的是一个二维的旋转矩阵
def get_rot(h):
    return torch.Tensor([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])


def img_transform(img, post_rot, post_tran,#post_rot, post_tran
                  resize, resize_dims, crop,
                  flip, rotate):
    #resize缩放比例、resize_dims缩放之后的目标尺寸、crop裁剪剩余区域、flip是否水平翻转、rotate数据增强时候的旋转角度
    # 图像空间变换部分
    # 调整图像尺寸 (例如 1600x900 -> 320x180)
    img = img.resize(resize_dims)
    # 随机/中心裁剪 (例如 320x180 裁剪出 352x128 区域)
    img = img.crop(crop)
    # 水平翻转（镜像）
    if flip:
        img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
    # 旋转图像（绕中心点旋转）
    img = img.rotate(rotate)

    # 几何变换矩阵更新部分
    # 缩放变换补偿（将缩放因子累积到旋转矩阵）
    post_rot *= resize
    # 裁剪变换补偿（减去裁剪起始点坐标）
    post_tran -= torch.Tensor(crop[:2])
    
    # 翻转变换补偿
    if flip:
        # 构造水平翻转矩阵 [[-1, 0], [0, 1]]
        A = torch.Tensor([[-1, 0], [0, 1]])
        # 计算翻转后的位置偏移
        b = torch.Tensor([crop[2] - crop[0], 0])
        # 更新旋转矩阵和平移向量
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b
    
    # 旋转变换补偿
    # 生成旋转矩阵（将角度转换为弧度）
    A = get_rot(rotate/180*np.pi)
    # 计算旋转中心偏移量（将裁剪区域中心对齐）
    b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
    # 调整旋转中心后的偏移补偿
    b = A.matmul(-b) + b
    # 更新旋转矩阵和平移向量
    post_rot = A.matmul(post_rot)
    post_tran = A.matmul(post_tran) + b

    return img, post_rot, post_tran

#反归一化，还原图像像素为原始的RGB
class NormalizeInverse(torchvision.transforms.Normalize):
    #  https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/8
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        #反归一化需要的计算参数
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)
    #把归一化之后的图像还原回去，并返回一个新的张量
    def __call__(self, tensor):
        return super().__call__(tensor.clone())

#把图像从标准状态恢复回来
denormalize_img = torchvision.transforms.Compose((
            NormalizeInverse(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
            torchvision.transforms.ToPILImage(),
        ))

#把图像归一化，相当于训练模型前面的预处理
normalize_img = torchvision.transforms.Compose((
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
))

#生成BEV特征图网格物理信息一些小格子的信息
def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])

    return dx, bx, nx
#dx代表每个方向多宽、bx代表第一个格子的中点在哪里、nx代表每个方向的格子数量

#合并相同位置的体素特征，就是对不同相机拍出来的重叠区域进行叠加防止出现一些失真等现象
def cumsum_trick(x, geom_feats, ranks):#x代表所有特征点的特征值，geom_feats特征点的几何位置，ranks每个点的排序编号
    x = x.cumsum(0)
    #标记哪些是新的体素点和之前不一样的部分
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    kept[:-1] = (ranks[1:] != ranks[:-1])
    #进行累加
    x, geom_feats = x[kept], geom_feats[kept]
    #还原为真值
    x = torch.cat((x[:1], x[1:] - x[:-1]))

    return x, geom_feats

#QuickCumsum 是 LSS 中用于 合并 BEV 网格中重叠点特征的高效实现，
class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None

#定义损失函数的形式
class SimpleLoss(torch.nn.Module):
    def __init__(self, pos_weight):
        super(SimpleLoss, self).__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]))

    def forward(self, ypred, ytgt):
        loss = self.loss_fn(ypred, ytgt)
        return loss

#计算每个 batch 中预测结果与标签之间的 IoU（交并比）
def get_batch_iou(preds, binimgs):
    """Assumes preds has NOT been sigmoided yet
    """
    with torch.no_grad():
        pred = (preds > 0)
        tgt = binimgs.bool()
        intersect = (pred & tgt).sum().float().item()
        union = (pred | tgt).sum().float().item()
    return intersect, union, intersect / union if (union > 0) else 1.0

#在验证集上评估模型性能，输出平均损失和IOU指标
def get_val_info(model, valloader, loss_fn, device, use_tqdm=False):
    model.eval()
    #累积的损失以及交、并集
    total_loss = 0.0
    total_intersect = 0.0
    total_union = 0
    print('running eval...')
    loader = tqdm(valloader) if use_tqdm else valloader
    with torch.no_grad():
        for batch in loader:
            allimgs, rots, trans, intrins, post_rots, post_trans, binimgs = batch
            preds = model(allimgs.to(device), rots.to(device),
                          trans.to(device), intrins.to(device), post_rots.to(device),
                          post_trans.to(device))
            binimgs = binimgs.to(device)

            # loss
            total_loss += loss_fn(preds, binimgs).item() * preds.shape[0]

            # iou
            intersect, union, _ = get_batch_iou(preds, binimgs)
            total_intersect += intersect
            total_union += union

    model.train()
    return {
            'loss': total_loss / len(valloader.dataset),
            'iou': total_intersect / total_union,
            }

#在BEV图上画出车辆
def add_ego(bx, dx):
    # approximate rear axel
    W = 1.85
    pts = np.array([
        [-4.084/2.+0.5, W/2.],
        [4.084/2.+0.5, W/2.],
        [4.084/2.+0.5, -W/2.],
        [-4.084/2.+0.5, -W/2.],
    ])
    pts = (pts - bx) / dx
    pts[:, [0,1]] = pts[:, [1,0]]
    plt.fill(pts[:, 0], pts[:, 1], '#76b900')

#加载所有的地图
def get_nusc_maps(map_folder):
    nusc_maps = {map_name: NuScenesMap(dataroot=map_folder,
                map_name=map_name) for map_name in [
                    "singapore-hollandvillage", 
                    "singapore-queenstown",
                    "boston-seaport",
                    "singapore-onenorth",
                ]}
    return nusc_maps

#在BEV图上面画地图
def plot_nusc_map(rec, nusc_maps, nusc, scene2map, dx, bx):
    egopose = nusc.get('ego_pose', nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
    map_name = scene2map[nusc.get('scene', rec['scene_token'])['name']]

    rot = Quaternion(egopose['rotation']).rotation_matrix
    rot = np.arctan2(rot[1, 0], rot[0, 0])
    center = np.array([egopose['translation'][0], egopose['translation'][1], np.cos(rot), np.sin(rot)])

    poly_names = ['road_segment', 'lane']
    line_names = ['road_divider', 'lane_divider']
    lmap = get_local_map(nusc_maps[map_name], center,
                         50.0, poly_names, line_names)
    for name in poly_names:
        for la in lmap[name]:
            pts = (la - bx) / dx
            plt.fill(pts[:, 1], pts[:, 0], c=(1.00, 0.50, 0.31), alpha=0.2)
    for la in lmap['road_divider']:
        pts = (la - bx) / dx
        plt.plot(pts[:, 1], pts[:, 0], c=(0.0, 0.0, 1.0), alpha=0.5)
    for la in lmap['lane_divider']:
        pts = (la - bx) / dx
        plt.plot(pts[:, 1], pts[:, 0], c=(159./255., 0.0, 1.0), alpha=0.5)

#提取地图信息
def get_local_map(nmap, center, stretch, layer_names, line_names):
    # need to get the map here...
    box_coords = (
        center[0] - stretch,
        center[1] - stretch,
        center[0] + stretch,
        center[1] + stretch,
    )

    polys = {}

    # polygons
    records_in_patch = nmap.get_records_in_patch(box_coords,
                                                 layer_names=layer_names,
                                                 mode='intersect')
    for layer_name in layer_names:
        polys[layer_name] = []
        for token in records_in_patch[layer_name]:
            poly_record = nmap.get(layer_name, token)
            if layer_name == 'drivable_area':
                polygon_tokens = poly_record['polygon_tokens']
            else:
                polygon_tokens = [poly_record['polygon_token']]

            for polygon_token in polygon_tokens:
                polygon = nmap.extract_polygon(polygon_token)
                polys[layer_name].append(np.array(polygon.exterior.xy).T)

    # lines
    for layer_name in line_names:
        polys[layer_name] = []
        for record in getattr(nmap, layer_name):
            token = record['token']

            line = nmap.extract_line(record['line_token'])
            if line.is_empty:  # Skip lines without nodes
                continue
            xs, ys = line.xy

            polys[layer_name].append(
                np.array([xs, ys]).T
                )

    # convert to local coordinates in place
    rot = get_rot(np.arctan2(center[3], center[2])).T
    for layer_name in polys:
        for rowi in range(len(polys[layer_name])):
            polys[layer_name][rowi] -= center[:2]
            polys[layer_name][rowi] = np.dot(polys[layer_name][rowi], rot)

    return polys
