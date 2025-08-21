## LSS数据管道解析
### 训练模式-数据流通管道
1. 初始传入参数
```
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
          )
```
2. 数据配置-创建数据加载器件
```
    trainloader, valloader = compile_data
    (version, dataroot,data_aug_conf=data_aug_conf,grid_conf=grid_conf,bsz=bsz,nworkers=nworkers,parser_name='segmentationdata'指定分割任务)
    对于compile_data（）函数
        a、初始化数据集：
            nusc = NuScenes(version='v1.0-{}'.format(version),#版本号dataroot=os.path.join(dataroot, version),#数据集路径verbose=False)
            （在这一步中会进入NuScenes SDK的_init_方法之后读取所有的json文件之后调用self._add_sample_data_index()得到
                sample = {
                    'token': 'abc123',
                    'timestamp': 1531971611981960,
                    'scene_token': 'scene_xyz',
                    'next': '...',
                    'prev': '...',
                    'data': {
                        'LIDAR_TOP': 'token_lidar_001',
                        'CAM_FRONT': 'token_cam_front_002',
                        ...
                    }
                }
                这样类型的例子）
        b、定义数据集traindata、valdata和数据加载器trainloader、valloader
        输入（version, dataroot,data_aug_conf=data_aug_conf,grid_conf=grid_conf,bsz=bsz,nworkers=nworkers,parser_name='segmentationdata'）
        （需要指定字段VizData或SegmentationData，
        其中VizData 返回 imgs, rots, trans, intrins, post_rots, post_trans, lidar_data, binimg
        SegmentationData 返回 imgs, rots, trans, intrins, post_rots, post_trans, binimg）
            （1）在这里会进行初始化调用NuscData（）
            （2）在NuscData（）会调用prepro（）函数进行samples的重新划分（根据训练类型选择对应的场景之后根据时间戳排序）返回samples
            （3）得到self.ixes = self.prepro()、rec = self.ixes[0]
            （4）调用
                get_image_data（）
                    sample_augmentation(H，W，'final_dim'，'resize_lim'，'rand_flip'，'rot_lim'，'bot_pct_lim')
                        函数数据增强的参数返回resize, resize_dims, crop, flip, rotate
                        （resize代表缩放比例，resize_dims代表缩放后的实际尺寸，crop代表矩形坐标，flip代表水平翻转，rotate代表旋转角度）（因为输入的是范围，在这一步进行随机选取翻转角度，缩放比例等）
                    img_transform(img, post_rot, post_tran，resize, resize_dims, crop,flip, rotate)
                        在这一步根据数据增强参数返回变换之后的img, post_rot, post_tran（图像和变换矩阵（从变换后到变换前的变换矩阵））
                choose_cams()
                get_binimg()加载返回带标签的雷达点云图（进行了坐标转换的投影到了BEV图内）
```
3. 模型、优化器、损失函数初始化
```
    model = compile_model(grid_conf, data_aug_conf, outC=1) 
            模型构建过程compile_model（）模型加载
                LiftSplatShoot
                    L：
                    create_frustum（）创建视锥体
                    get_geometry（）将视锥体坐标变换到自车坐标系
                    get_cam_feats（）提取特征以及深度概率分布
                            camencode（）
                                get_depth_feat()两个功能：特征提取+预测深度分布和加权特征图
                                    get_eff_depth（）特征提取（从EfficientNet提取特征之后通过UP（）进行特征融合）
                                    get_depth_dist（）预测深度分布
                    S：
                        voxel_pooling（）将特征投影到BEV网格（展评、过滤、排序、求和）
                    bevencode（）将BEV特征图输出为BEV语义图
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = SimpleLoss(pos_weight).cuda(gpuid) if gpuid >=0 else SimpleLoss(pos_weight)
```
4. 主训练循环
```
        （1）训练流程
        for epoch in range(nepochs):
        np.random.seed()
        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, binimgs) in enumerate(trainloader):
            t0 = time()
            前向传播、计算损失、反向传播与优化、更新计数器、日志记录与验证
                前向传播
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
        （2）模型的输入、输出总结（B代表batch size）
                输入：
                    imgs：输入图像（B，C，H，W）
                    rots：相机旋转矩阵（B，4，4）
                    trans：相机平移向量（B，3）
                    intrins：相机内参矩阵（B，3，3）
                    post_rots：后处理旋转矩阵（B，4，4）
                    post_trans：后处理平移向量（B，3）
                输出：
                    预测的BEV语义图（B，C，H，W）
                    （C：类别数/通道数，H×W：空间分辨率）
        （3）计算损失
            loss = loss_fn(preds, binimgs)
                        损失函数的设计
                        
                    （1）BEV 标签图 binimgs的结构：binimgs.shape = [B, C, H, W]  # B: batch size, C: 类别数/通道数，H×W: 空间分辨网格
                    （2）损失函数类型：BCEWithLogitsLoss：适用于多标签分割，能够自动将 logits 通过 sigmoid 转换为概率，然后进行二分类交叉熵计算
                    （3）数学公式  权重+log的损失计算
                            L=-w*y*log(o(x))-(1-y)*log(1-o(x))
                            Total Loss=L求和/B*C*H*W（多通道（多个检测对象时候）求平均值，单个通道时候不需要求平均）
                            x预测值、y真实标签、w正类权重、o（x）sigmoid函数
                    （4）代码实现
                            class SimpleLoss(torch.nn.Module):
                            def __init__(self, pos_weight):
                                super(SimpleLoss, self).__init__()
                                self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]))
                            def forward(self, ypred, ytgt):
                                loss = self.loss_fn(ypred, ytgt)
                                return loss
                    （5）模型训练时候的应用
                            binimgs = binimgs.to(device)
                            loss = loss_fn(preds, binimgs)

                            loss.backward()
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                                        opt.step()
        （4）反向传播与优化
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
```





















