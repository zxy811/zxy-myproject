## ONNX可视化模型结构
```
运行根目录下convert_lss_to_onnx.py文件
```
## 第1层：体素池化初始化层
```
体素池化 
主要操作类型：
对应代码部分：
def create_frustum(self):
    ogfH, ogfW = self.data_aug_conf['final_dim']  # (128, 352)
    fH, fW = ogfH // self.downsample, ogfW // self.downsample  # (8, 22)
    ds = torch.arange(*self.grid_conf['dbound']).view(-1, 1, 1).expand(-1, fH, fW)
    # 创建frustum网格: [D, H, W, 3]
```

## 第2层：几何变换处理层
```
处理层 - 第2层 
主要功能：几何坐标变换、这些层执行复杂的坐标系变换，将每个相机的frustum特征从相机坐标系转换到统一的自车坐标系
def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
    """将frustum点云转换到自车坐标系"""
    B, N, _ = trans.shape   
    # 撤销后处理变换
    points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
    points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))        
    # 相机坐标到自车坐标变换
    points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                        points[:, :, :, :, :, 2:3]), 5)
    combine = rots.matmul(torch.inverse(intrins))
    points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
    points += trans.view(B, N, 1, 1, 1, 3)
```

## 第3层：相机编码器
```
主要功能：特征提升提取、为每个像素预测深度分布概率，然后将图像特征与深度概率相乘，生成带有深度信息的3D特征
Conv + Sigmoid : EfficientNet特征提取  获取深度和特征
Shape/Gather/Sub : 继续进行张量形状计算
Cast/Reshape : 数据类型转换和重塑
对应代码部分：
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
```

## 第4层：BEV编码器核心层（501-701）
```
主要功能：Splat操作 - 3D到BEV投影、将来自多个相机的3D特征"拍扁"到统一的BEV网格中。使用累加和技巧（cumsum trick）高效地将重叠的特征点融合（结构图使用Reshape-堆叠 + 零布占位）
关键操作：
大量Conv + Sigmoid + Mul: ResNet特征提取
ConstantOfShape + Concat: 创建BEV网格
GlobalAveragePool: 特征池化
对应代码部分：
    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B*N*D*H*W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx/2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime//B, 1], ix,
                             device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])\
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])\
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)\
            + geom_feats[:, 1] * (self.nx[2] * B)\
            + geom_feats[:, 2] * B\
            + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)

        return final

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        x = self.get_cam_feats(x)

        x = self.voxel_pooling(geom, x)

        return x
```   

## 第5层：BEV后处理层
```
主要功能：BEV特征编码、在BEV特征图上进行语义分割或目标检测等下游任务。使用ResNet架构提取语义特征，最终输出用于自动驾驶的BEV感知结果
Unsqueeze: 维度扩展
Conv + Relu: ResNet编码
Split + Squeeze: 张量分割和压缩
第16层操作：
Conv + Relu: 最终特征提取
Resize : 上采样到目标分辨率
对应代码部分：
    class BevEncode(nn.Module):
        def __init__(self, inC, outC):
            trunk = resnet18(pretrained=False, zero_init_residual=True)
            self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.layer1 = trunk.layer1  # ResNet层
            self.layer2 = trunk.layer2
            self.layer3 = trunk.layer3
            self.up1 = Up(64+256, 256, scale_factor=4)  # 上采样
            self.up2 = nn.Sequential(...)  # 最终输出层
            
        def forward(self, x):
            x = self.conv1(x)
            x1 = self.layer1(x)
            x = self.layer2(x1)
            x = self.layer3(x)
            x = self.up1(x, x1)  # 特征融合
            x = self.up2(x)      # 最终输出
            return x
 
```
## 第6层：LSS模型数据流总结
```
LSS模型数据流总结
   数量
    ↓  
几何坐标变换 (相机系→自车系)
    ↓
EfficientNet特征提取 + 深度估计 (Lift)
    ↓
Voxel Pooling + BEV投影 (Splat)  
    ↓
ResNet BEV特征编码 (Shoot)
    ↓
bev_output [dynamic,1,dynamic,dynamic] (BEV语义图)
```
```
1、首先这个模型的输入是原始图像信息，以及一些图像增强的参数，输出是BEV预测图，其中的流程首先是3D视锥体的创建，这个创建实际上就是根据【1，25，0.1】类似于这样生成了XYZ三个轴上面的小格子这一步不需要图像的信息，只需要维度长度以及步长，这些都会输入

2、之后就是进行了几何变换层，就是把这个3D视锥体首先撤销后处理变换以及将相机坐标转换到自车坐标系（从像素到相机到车体坐标系）

3、再之后就是相机的特征提取部分，主要进行的操作是输入图像，之后这个部分的主干网络是使用预训练的EfficientNet-b0（这个主要是进行各个conv之后拼接然后conv）进行特征提取和融合（只进行图片的特征提取），之后经过一层卷积来预测深度，后经过softmax获得深度概率分布，之后将深度和特征进行分离之后将特征进行概率加权处理

4、之后就是展平的阶段，展平的逻辑首先是先对get_geometry这个输出的3D点进行处理，就是将原点对齐在格子的中点，之后➗dx得到相当于是格子级别的索引，之后改变维度加上batch的索引，变成（nprime,4）这样的类型，之后就是过滤掉一些点之后进行分配rank之后根据rank进行排序求和做差，因为每一个格子里面可能会有多个点，这样的话就得到每一个格子里面对应的特征的信息，之后定义一个3D的BEV网络图，再将上面我们得到的信息填入进去，之后进行Z轴上面的切分，将上面的3D体按照z轴上面的维度进行切分之后在通道层面进行拼接，到这里拍扁这个过程就结束了

5、然后最后送到BevEncode里面经过卷积以及其他的层或者是块的操作来最终得到我们的BEV预测图
```