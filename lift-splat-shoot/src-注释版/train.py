# 导入必要的库
import torch
from time import time
from tensorboardX import SummaryWriter  # 训练可视化工具
import numpy as np
import os

# 从本地模块导入组件
from .models import compile_model  # 模型构建函数
from .data import compile_data     # 数据加载函数
from .tools import SimpleLoss, get_batch_iou, get_val_info  # 损失函数和评估指标

# 训练主函数定义
def train(version,                # 数据集版本 (mini/trainval)
          dataroot='/data/nuscenes',  # 数据集根目录
          nepochs=10000,          # 最大训练epoch数
          gpuid=1,                # 使用的GPU ID

          # 图像预处理参数
          H=900, W=1600,          # 原始图像分辨率
          resize_lim=(0.193, 0.225),  # 随机缩放范围
          final_dim=(128, 352),   # 输出尺寸 (高,宽)
          bot_pct_lim=(0.0, 0.22),# 底部裁剪比例
          rot_lim=(-5.4, 5.4),    # 随机旋转角度范围(度)
          rand_flip=True,         # 启用随机水平翻转
          ncams=5,                # 使用的摄像头数量

          # 训练参数
          max_grad_norm=5.0,      # 梯度裁剪阈值
          pos_weight=2.13,       # 正样本权重（用于类别不平衡）
          logdir='./runs',        # 日志保存目录

          # 空间网格参数
          xbound=[-50.0, 50.0, 0.5],  # X轴划分 [-50m,50m] 0.5m网格
          ybound=[-50.0, 50.0, 0.5],  # Y轴划分
          zbound=[-10.0, 10.0, 20.0], # Z轴划分
          dbound=[4.0, 45.0, 1.0],    # 深度划分

          # 数据加载配置
          bsz=4,                  # 批大小
          nworkers=10,            # 数据加载线程数
          
          # 优化器参数
          lr=1e-3,                # 学习率
          weight_decay=1e-7,      # 权重衰减
          ):
    
    # 空间网格配置
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    
    # 数据增强配置
    data_aug_conf = {
        'resize_lim': resize_lim,
        'final_dim': final_dim,
        'rot_lim': rot_lim,
        'H': H, 'W': W,
        'rand_flip': rand_flip,
        'bot_pct_lim': bot_pct_lim,
        'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
        'Ncams': ncams,  # 实际使用的摄像头数
    }

    # 创建数据加载器（使用分割数据解析器）
    trainloader, valloader = compile_data(
        version, dataroot,
        data_aug_conf=data_aug_conf,
        grid_conf=grid_conf,
        bsz=bsz,
        nworkers=nworkers,
        parser_name='segmentationdata'  # 指定分割任务
    )

    # 设置计算设备
    device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')

    # 初始化模型并移至设备
    model = compile_model(grid_conf, data_aug_conf, outC=1)  # 输出通道1（二值分割）
    model.to(device)

    # 定义Adam优化器
    opt = torch.optim.Adam(model.parameters(), 
                         lr=lr, 
                         weight_decay=weight_decay)

    # 初始化损失函数（带类别平衡权重）
    loss_fn = SimpleLoss(pos_weight).cuda(gpuid) if gpuid >=0 else SimpleLoss(pos_weight)

    # 初始化TensorBoard记录器
    writer = SummaryWriter(logdir=logdir)
    
    # 设置验证间隔（mini数据集每1000步验证，完整数据集每10000步）
    val_step = 1000 if version == 'mini' else 10000

    # 主训练循环
    model.train()
    counter = 0  # 全局步数计数器
    for epoch in range(nepochs):
        np.random.seed()  # 每个epoch重置随机种子
        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, binimgs) in enumerate(trainloader):
            t0 = time()
            
            # 梯度清零
            opt.zero_grad()
            
            # 前向传播
            preds = model(
                imgs.to(device),
                rots.to(device),
                trans.to(device),
                intrins.to(device),
                post_rots.to(device),
                post_trans.to(device),
            )
            
            # 计算损失
            binimgs = binimgs.to(device)
            loss = loss_fn(preds, binimgs)
            
            # 反向传播与优化
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # 梯度裁剪
            opt.step()
            
            # 更新计数器
            counter += 1
            t1 = time()

            # 日志记录（每10步记录损失）
            if counter % 10 == 0:
                print(f'Step: {counter}, Loss: {loss.item():.4f}')
                writer.add_scalar('train/loss', loss, counter)
            
            # 扩展日志记录（每50步记录IoU和耗时）
            if counter % 50 == 0:
                _, _, iou = get_batch_iou(preds, binimgs)
                writer.add_scalar('train/iou', iou, counter)
                writer.add_scalar('train/epoch', epoch, counter)
                writer.add_scalar('train/step_time', t1 - t0, counter)
            
            # 验证与保存模型（按val_step间隔）
            if counter % val_step == 0:
                # 在验证集上评估
                val_info = get_val_info(model, valloader, loss_fn, device)
                print(f'Validation - Loss: {val_info["loss"]:.4f}, IoU: {val_info["iou"]:.4f}')
                writer.add_scalar('val/loss', val_info['loss'], counter)
                writer.add_scalar('val/iou', val_info['iou'], counter)
                
                # 保存模型检查点
                model.eval()
                mname = os.path.join(logdir, f"model{counter}.pt")
                torch.save(model.state_dict(), mname)
                model.train()
