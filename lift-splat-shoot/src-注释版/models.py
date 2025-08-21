"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18

from .tools import gen_dx_bx, cumsum_trick, QuickCumsum


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        # 上采样层：双线性插值
        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        
        # 卷积模块：包含两个卷积层
        self.conv = nn.Sequential(
            # 第一个卷积：3x3卷积 + BN + ReLU
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # 第二个卷积：3x3卷积 + BN + ReLU
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)  # 上采样输入特征x1
        x1 = torch.cat([x2, x1], dim=1)  # 与跳跃连接x2在通道维度拼接
        return self.conv(x1)  # 通过卷积模块


class CamEncode(nn.Module):
    def __init__(self, D, C, downsample):
        super(CamEncode, self).__init__()
        self.D = D  # 深度通道数
        self.C = C  # 特征通道数
        
        # 主干网络：使用预训练的EfficientNet-b0
        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")
        
        # 特征融合模块：将不同尺度的特征进行融合
        self.up1 = Up(320+112, 512)  # 输入320+112通道，输出512通道
        
        # 深度和特征预测头
        self.depthnet = nn.Conv2d(512, self.D + self.C, kernel_size=1)  # 1x1卷积

    def get_depth_dist(self, x):
        return x.softmax(dim=1)  # 深度概率分布

    def get_depth_feat(self, x):
        x = self.get_eff_depth(x)  # 获取多尺度特征
        x = self.depthnet(x)  # 预测深度+特征
        
        # 分割深度预测和特征
        depth = self.get_depth_dist(x[:, :self.D])  # 深度概率 [B,D,H,W]
        new_x = depth.unsqueeze(1) * x[:, self.D:].unsqueeze(2)  # 特征加权 [B,C,D,H,W]
        return depth, new_x

    def get_eff_depth(self, x):
        # 提取EfficientNet的多层特征
        endpoints = {}
        # Stem层（初始卷积）
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x
        
        # 遍历所有Block
        for idx, block in enumerate(self.trunk._blocks):
            # 应用带DropPath的Block
            x = block(x, drop_connect_rate=...)
            
            # 记录下采样前的特征
            if prev_x.size(2) > x.size(2):
                endpoints[f'reduction_{len(endpoints)+1}'] = prev_x
            prev_x = x
        
        # 融合最后两个层级的特征
        endpoints['reduction_5'] = x
        return self.up1(endpoints['reduction_5'], endpoints['reduction_4'])  # 特征融合

    def forward(self, x):
        depth, x = self.get_depth_feat(x)  # 获取深度和加权特征
        return x  # 输出形状 [B, C, D, H, W]


class BevEncode(nn.Module):
    def __init__(self, inC, outC):
        super(BevEncode, self).__init__()
        
        # 基于ResNet18的骨干网络
        trunk = resnet18(pretrained=False, zero_init_residual=True)
        
        # 自定义第一层（适配输入通道）
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = trunk.bn1     # 继承ResNet的BN层
        self.relu = trunk.relu   # 继承ReLU
        
        # 使用ResNet的中间层
        self.layer1 = trunk.layer1  # 第一组残差块
        self.layer2 = trunk.layer2  # 第二组残差块
        self.layer3 = trunk.layer3  # 第三组残差块
        #layer1: [B, 64, H/2, W/2]
        #layer2: [B, 128, H/4, W/4]
        #layer3: [B, 256, H/8, W/8]

        # 上采样模块
        self.up1 = Up(64+256, 256, scale_factor=4)  # 跳跃连接融合
        self.up2 = nn.Sequential(  # 最终上采样头
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, outC, 1)  # 输出预测
        )

    def forward(self, x):
        # 初始卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # 残差块处理
        x1 = self.layer1(x)  # 保存跳跃连接特征
        x = self.layer2(x1)
        x = self.layer3(x)
        
        # 上采样过程
        x = self.up1(x, x1)  # 融合layer3和layer1的特征
        x = self.up2(x)       # 最终上采样到目标尺寸
        return x


class LiftSplatShoot(nn.Module):
    def __init__(self, grid_conf, data_aug_conf, outC):
        #outc是最终输出的通道数
        super(LiftSplatShoot, self).__init__()#初始化
        # 初始化网格配置和数据增强配置
        self.grid_conf = grid_conf
        self.data_aug_conf = data_aug_conf

        # 生成网格参数（dx: 网格尺寸，bx: 网格起点，nx: 网格数量）
        dx, bx, nx = gen_dx_bx(
            self.grid_conf['xbound'],
            self.grid_conf['ybound'],
            self.grid_conf['zbound'],
        )
        # 将网格参数注册为不可训练参数，避免训练中网格出现变形
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        # 设置下采样率和相机特征通道数
        self.downsample = 16
        self.camC = 64
        
        # 创建视锥体（图像平面到3D空间的映射网格）
        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape  # 获取深度维度数
        
        # 初始化相机编码器和BEV编码器
        self.camencode = CamEncode(self.D, self.camC, self.downsample)
        self.bevencode = BevEncode(inC=self.camC, outC=outC)

        # 选择是否使用快速累积求和
        self.use_quickcumsum = True

    def create_frustum(self):
        #创建图像平面到3D空间的映射网格，创建3D视锥
        #获取原始图像尺寸和下采样后尺寸
        ogfH, ogfW = self.data_aug_conf['final_dim']
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        
        # 生成深度序列（dbound来自网格配置）
        ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float)
        ds = ds.view(-1, 1, 1).expand(-1, fH, fW)  # 扩展为[D, H, W]
        
        # 生成x坐标网格（0到图像宽度）
        xs = torch.linspace(0, ogfW - 1, fW)
        xs = xs.view(1, 1, fW).expand(len(ds), fH, fW)
        
        # 生成y坐标网格（0到图像高度）
        ys = torch.linspace(0, ogfH - 1, fH)
        ys = ys.view(1, fH, 1).expand(len(ds), fH, fW)
        
        # 组合成3D网格 [D, H, W, 3]
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)#转化为模型参数不参与训练

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """计算3D点在自车坐标系中的位置"""
        B, N, _ = trans.shape  # Batch大小，摄像头数量
        
        # 1. 撤销数据增强的变换（平移+旋转）
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3) @ points.unsqueeze(-1)
        
        # 2. 转换到相机坐标系（去畸变）
        points = torch.cat((
            points[..., :2] * points[..., 2:3],  # x = x'*z, y = y'*z
            points[..., 2:3]                     # z保持原值
        ), dim=5)
        
        # 3. 转换到自车坐标系（应用外参）
        combine = rots @ torch.inverse(intrins)  # 组合变换矩阵
        points = combine.view(B, N, 1, 1, 1, 3, 3) @ points.unsqueeze(-1)
        points = points.squeeze(-1) + trans.view(B, N, 1, 1, 1, 3)
        
        return points

    def get_cam_feats(self, x):
        """提取相机特征"""
        B, N, C, imH, imW = x.shape  # 输入形状
        
        # 重塑输入并提取特征
        x = x.view(B*N, C, imH, imW)
        x = self.camencode(x)  # 输出形状 [B*N, C, D, fH, fW]
        
        # 调整维度顺序
        x = x.view(B, N, self.camC, self.D, imH//self.downsample, imW//self.downsample)
        return x.permute(0, 1, 3, 4, 5, 2)  # [B, N, D, H, W, C]

    def voxel_pooling(self, geom_feats, x):
        """体素池化：将特征投影到BEV网格"""
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W  # 总点数
        
        # 展平特征和坐标
        x = x.reshape(Nprime, C)
        geom_feats = ((geom_feats - (self.bx - self.dx/2)) / self.dx).long()
        
        # 添加batch索引
        batch_ix = torch.cat([torch.full([Nprime//B, 1], i) for i in range(B)])
        geom_feats = torch.cat([geom_feats.view(-1, 3), batch_ix], 1)
        
        # 过滤超出范围的坐标
        mask = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) 
             & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])
             & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x, geom_feats = x[mask], geom_feats[mask]
        
        # 排序和累积求和
        ranks = geom_feats[:, 0] * (self.nx[1]*self.nx[2]*B) 
               + geom_feats[:, 1] * (self.nx[2]*B) 
               + geom_feats[:, 2] * B 
               + geom_feats[:, 3]
        indices = ranks.argsort()
        
        # 使用快速累积或普通方法
        if self.use_quickcumsum:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)
        else:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        
        # 构建BEV特征图
        final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x
        
        # 合并Z轴
        return torch.cat(final.unbind(dim=2), 1)

    def forward(self, x, rots, trans, intrins, post_rots, post_trans):
        """完整前向传播流程"""
        # 1. 计算3D几何坐标
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        
        # 2. 提取相机特征
        x = self.get_cam_feats(x)
        
        # 3. 体素池化得到BEV特征
        x = self.voxel_pooling(geom, x)
        
        # 4. BEV编码输出最终结果
        return self.bevencode(x)


def compile_model(grid_conf, data_aug_conf, outC):
    return LiftSplatShoot(grid_conf, data_aug_conf, outC)
