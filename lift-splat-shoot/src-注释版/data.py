"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
import os
import numpy as np
from PIL import Image
import cv2
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import Box
from glob import glob

from .tools import get_lidar_data, img_transform, normalize_img, gen_dx_bx


class NuscData(torch.utils.data.Dataset):
    def __init__(self, nusc, is_train, data_aug_conf, grid_conf):
        self.nusc = nusc
        self.is_train = is_train
        self.data_aug_conf = data_aug_conf
        self.grid_conf = grid_conf
        self.scenes = self.get_scenes()
        self.ixes = self.prepro()
        dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()
        self.fix_nuscenes_formatting()
        print(self)
    def fix_nuscenes_formatting(self):
        #调整数据集文件路径格式
        """If nuscenes is stored with trainval/1 trainval/2 ... structure, adjust the file paths
        stored in the nuScenes object.
        """
        # check if default file paths work
        rec = self.ixes[0]
        sampimg = self.nusc.get('sample_data', rec['data']['CAM_FRONT'])
        imgname = os.path.join(self.nusc.dataroot, sampimg['filename'])
        #定义路径解析函数
        def find_name(f):
            
            d, fi = os.path.split(f)
            d, di = os.path.split(d)
            d, d0 = os.path.split(d)
            d, d1 = os.path.split(d)
            d, d2 = os.path.split(d)
            return di, fi, f'{d2}/{d1}/{d0}/{di}/{fi}'
        #如果默认路径无效，重新映射文件路径
        # adjust the image paths if needed
        if not os.path.isfile(imgname):
            print('adjusting nuscenes file paths')
            fs = glob(os.path.join(self.nusc.dataroot, 'samples/*/samples/CAM*/*.jpg'))
            fs += glob(os.path.join(self.nusc.dataroot, 'samples/*/samples/LIDAR_TOP/*.pcd.bin'))
            info = {}
            for f in fs:
                di, fi, fname = find_name(f)
                info[f'samples/{di}/{fi}'] = fname
            fs = glob(os.path.join(self.nusc.dataroot, 'sweeps/*/sweeps/LIDAR_TOP/*.pcd.bin'))
            for f in fs:
                di, fi, fname = find_name(f)
                info[f'sweeps/{di}/{fi}'] = fname
            for rec in self.nusc.sample_data:
                if rec['channel'] == 'LIDAR_TOP' or (rec['is_key_frame'] and rec['channel'] in self.data_aug_conf['cams']):
                    rec['filename'] = info[rec['filename']]

    
    #官方提供函数进行场景的划分
    def get_scenes(self):
        # filter by scene split
        split = {
            'v1.0-trainval': {True: 'train', False: 'val'},
            'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
        }[self.nusc.version][self.is_train]

        scenes = create_splits_scenes()[split]#官方划分形式

        return scenes
    #获取所有的样本、之后过滤不属于当前划分的场景的样本、最后进行帧、和时间戳排序
    def prepro(self):
        samples = [samp for samp in self.nusc.sample]

        # remove samples that aren't in this split
        samples = [samp for samp in samples if
                   self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes]

        # sort by scene, timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

        return samples
    #数据增强的核心实现方法，获得数据增强参数
    def sample_augmentation(self):
        H, W = self.data_aug_conf['H'], self.data_aug_conf['W']
        fH, fW = self.data_aug_conf['final_dim']
        if self.is_train:
            #随机缩放
            resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            #随机裁剪告诉我ndom.uniform(*self.data_aug_conf['bot_pct_lim']))*newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            #水平翻转
            flip = False
            if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            #随机旋转
            rotate = np.random.uniform(*self.data_aug_conf['rot_lim'])
        else:
            #计算最小缩放比例
            resize = max(fH/H, fW/W)
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            #中心裁剪
            crop_h = int((1 - np.mean(self.data_aug_conf['bot_pct_lim']))*newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate
        #resize代表缩放比例，resize_dims代表缩放后的实际尺寸，crop代表矩形坐标，flip代表水平翻转，rotate代表旋转角度
    def get_image_data(self, rec, cams):
        # 初始化存储各摄像头数据的列表
        imgs = []       # 图像像素数据
        rots = []       # 相机旋转矩阵
        trans = []      # 相机平移向量
        intrins = []    # 相机内参矩阵
        post_rots = []  # 数据增强后的旋转补偿矩阵
        post_trans = [] # 数据增强后的平移补偿向量
        
        # 遍历每个摄像头
        for cam in cams:
            # 获取摄像头数据
            samp = self.nusc.get('sample_data', rec['data'][cam])
            imgname = os.path.join(self.nusc.dataroot, samp['filename'])
            img = Image.open(imgname)  # 加载原始图像
            
            # 初始化后处理矩阵（用于补偿数据增强带来的几何变化）
            post_rot = torch.eye(2)   # 2x2单位矩阵
            post_tran = torch.zeros(2) # 零平移
        
            # 获取相机标定参数
            sens = self.nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])
            intrin = torch.Tensor(sens['camera_intrinsic'])  # 内参矩阵 [[fx,0,cx],[0,fy,cy],[0,0,1]]
            rot = torch.Tensor(Quaternion(sens['rotation']).rotation_matrix)  # 相机到车体的旋转
            tran = torch.Tensor(sens['translation'])  # 相机到车体的平移
        
            # 数据增强变换
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation()
            img, post_rot2, post_tran2 = img_transform(
                img, post_rot, post_tran,
                resize=resize,       # 缩放比例
                resize_dims=resize_dims, # 缩放后尺寸
                crop=crop,           # 裁剪区域
                flip=flip,           # 是否水平翻转
                rotate=rotate        # 旋转角度
            )
            
            # 将2x2变换矩阵扩展为3x3齐次坐标矩阵
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2  # 平移分量
            post_rot[:2, :2] = post_rot2  # 旋转分量
        
            # 收集各摄像头数据
            imgs.append(normalize_img(img))  # 归一化图像 [0,255] -> [0,1]
            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)  # 增强后的补偿旋转
            post_trans.append(post_tran) # 增强后的补偿平移
        
        # 将列表转换为张量堆叠
        return (torch.stack(imgs), torch.stack(rots), torch.stack(trans),
                torch.stack(intrins), torch.stack(post_rots), torch.stack(post_trans))
    #提取单帧或者是融合之后多帧的激光雷达点云数据
    def get_lidar_data(self, rec, nsweeps):
        pts = get_lidar_data(self.nusc, rec,
                       nsweeps=nsweeps, min_distance=2.2)
        return torch.Tensor(pts)[:3]  # x,y,z#用于后续与图像进行投影对齐

    def get_binimg(self, rec):
        #获取激光雷达之后转到自车坐标系
        egopose = self.nusc.get('ego_pose',
                                self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        trans = -np.array(egopose['translation'])
        rot = Quaternion(egopose['rotation']).inverse
        #创建一个空的BEV图
        img = np.zeros((self.nx[0], self.nx[1]))
        #遍历当前帧的所有的标注框
        for tok in rec['anns']:
            inst = self.nusc.get('sample_annotation', tok)
            # add category for lyft
            if not inst['category_name'].split('.')[0] == 'vehicle':
                continue
            #将3Dbox从世界坐标转移到自车坐标系
            box = Box(inst['translation'], inst['size'], Quaternion(inst['rotation']))
            box.translate(trans)
            box.rotate(rot)
            #将识别出来的从自车坐标系转移到BEV网格中的坐标
            pts = box.bottom_corners()[:2].T
            pts = np.round(
                (pts - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]
                ).astype(np.int32)
            pts[:, [1, 0]] = pts[:, [0, 1]]
            #进行填充
            cv2.fillPoly(img, [pts], 1.0)

        return torch.Tensor(img).unsqueeze(0)

    #用于选择相机，如果是训练阶段则选择出来部分相机，如果是验证阶段则选择所有相机
    def choose_cams(self):
        if self.is_train and self.data_aug_conf['Ncams'] < len(self.data_aug_conf['cams']):
            cams = np.random.choice(self.data_aug_conf['cams'], self.data_aug_conf['Ncams'],
                                    replace=False)
        else:
            cams = self.data_aug_conf['cams']
        return cams
    #用于返回数据集的基本信息
    def __str__(self):
        return f"""NuscData: {len(self)} samples. Split: {"train" if self.is_train else "val"}.
                   Augmentation Conf: {self.data_aug_conf}"""
    #返回数据集的长度也就是样本的数量
    def __len__(self):
        return len(self.ixes)

#可视化数据的加载，返回各种数据可作为后续网络训练的输入，有激光雷达数据
class VizData(NuscData):
    def __init__(self, *args, **kwargs):
        super(VizData, self).__init__(*args, **kwargs)
    
    def __getitem__(self, index):
        rec = self.ixes[index]
        rec = self.ixes[inlll]
        
        cams = self.choose_cams()
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(rec, cams)
        lidar_data = self.get_lidar_data(rec, nsweeps=3)
        binimg = self.get_binimg(rec)
        
        return imgs, rots, trans, intrins, post_rots, post_trans, lidar_data, binimg
        #binimg语义标签图（BEV 标签）

#用于语义分割任务的数据集
class SegmentationData(NuscData):
    def __init__(self, *args, **kwargs):
        super(SegmentationData, self).__init__(*args, **kwargs)
    
    def __getitem__(self, index):
        rec = self.ixes[index]

        cams = self.choose_cams()
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(rec, cams)
        binimg = self.get_binimg(rec)
        
        return imgs, rots, trans, intrins, post_rots, post_trans, binimg


def worker_rnd_init(x):
    np.random.seed(13 + x)

#创建选择数据加载的方式
def compile_data(version, dataroot, data_aug_conf, grid_conf, bsz,
                 nworkers, parser_name):
    #初始化数据集
    nusc = NuScenes(version='v1.0-{}'.format(version),#版本号
                    dataroot=os.path.join(dataroot, version),#数据集路径
                    verbose=False)
    #数据解析器类型
    parser = {
        'vizdata': VizData,
        'segmentationdata': SegmentationData,
    }[parser_name]
    #定义数据集
    traindata = parser(nusc, 
                       is_train=True, 
                       data_aug_conf=data_aug_conf,#传入数据增强配置
                         grid_conf=grid_conf)
    valdata = parser(nusc, 
                     is_train=False, 
                     data_aug_conf=data_aug_conf,
                       grid_conf=grid_conf)
    #定义数据加载器
    trainloader = torch.utils.data.DataLoader(traindata, batch_size=bsz,
                                              shuffle=True,#训练时打乱顺序
                                              num_workers=nworkers,#并行加载进程数
                                              drop_last=True,#丢弃最后不完整batch
                                              worker_init_fn=worker_rnd_init)
    valloader = torch.utils.data.DataLoader(valdata, batch_size=bsz,
                                            shuffle=False,#验证时打乱顺序
                                            num_workers=nworkers)

    return trainloader, valloader 