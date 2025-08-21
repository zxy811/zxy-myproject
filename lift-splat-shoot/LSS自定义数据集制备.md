## 自定义nuScenes格式数据集指南
### 参考资料
```
1、nuScenes数据集下载[https://www.nuscenes.org/](https://www.nuscenes.org/).
2、nuScenes数据集格式调整参考
      CSDN：https://blog.csdn.net/zyw2002/article/details/128319169
```
### nuscenes 主要json文件数据结构解析
```
calibrated_sensor.json
        calibrated_sensor {
        "token":                   唯一的标识符
        "sensor_token":            指向传感器类型的外键。
        "translation":             以米为单位的坐标系原点: x, y, z.
        "rotation":                坐标系方向为四元数: w, x, y, z.
        "camera_intrinsic":        内置相机校准。对于非相机的传感器为空。
        }
sample.json：
        sample {
        "token":                   唯一的标识符.
        "timestamp":               时间戳
        "scene_token":             指向scene的
        "next":                    下一帧的token
        "prev":                    上一帧的token
        }
sample_annotation
        sample_annotation {
        "token":                   唯一的标识符
        "sample_token":            注释属于哪一帧（而不是哪一张图片）
        "instance_token":          同一个物体在整个序列（例如一辆车、一棵树）用同一个 instance_token
        "attribute_tokens":        此注释的属性列表
        "visibility_token":        能见度也可能随着时间的推移而改变。如果没有标注可见性，则token为空字符串。
        "translation":             边界框的位置，以米为单位，如center_x, center_y, center_z
        "size":                    边框大小以米为单位，如宽、长、高。
        "rotation":                边框方向为四元数:w, x, y, z
        "num_lidar_pts":           这个检测框里的激光雷达点的数量
        "num_radar_pts":           这个盒子里雷达点的数目
        "next":                    
        "prev":                    
        }
sample_data
        sample_data {
        "token":                   唯一的标识符.
        "sample_token":            与此sample_data相关联的示例。
        "ego_pose_token":          
        "calibrated_sensor_token": 
        "filename":                磁盘上data-blob的相对路径.
        "fileformat":              数据文件格式.
        "width":                   如果样本数据是一张图像，则这是图像宽度(以像素为单位)
        "height":                  如果样本数据是一张图像，则这是图像高度(以像素为单位)。
        "timestamp":               Unix时间戳。
        "is_key_frame":            如果sample_data是key_frame的一部分，则为True，否则为False。
        "next":                    
        "prev":                    
        }
scene
        scene {
        "token":                   唯一的标识符.
        "name":                    场景名称.
        "description":             场景描述.
        "log_tokens":              场景所属的日志文件列表.
        "nbr_samples":             场景中的样本数量.
        "first_sample_token":      场景中的第一个样本的token.
        "last_sample_token":       场景中的最后一个样本的token.
        }
sensor
        sensor {
        "token":                   唯一的标识符.
        "channel":                 传感器通道名称
        "modality":                {camera, lidar, radar} -- 传感器形态。支持括号内的类别
        }
category.json
        category {
        "token":                   唯一的标识符.
        "name":                    类别名称.
        "description":             类别描述.
        }
```
### 自定义nuScenes格式数据集指南
```
    samples
        六个相机图片输入 、一个LIDAR——TOP（后续需要打标签）
    v1.0-mini
        sample.json：                   （脚本）样本时间帧序列主入口，模型训练按它加载每一帧（有前后token）
        sample_data.json：              （脚本）通过它获取图像路径、时间戳、sensor_id、pose_token
        calibrated_sensor.json：        （自定义）获取相机内参（K矩阵）+ 外参（T_cam_to_ego）
        sample_annotation.json：        （脚本）每一帧的 3D 物体标注
        sensor.json:                    （自定义）每个传感器的类型、名称等静态信息
        category.json                   （自定义）定义所有目标类别 ID → 名称
        scene.json：                    （自定义）每个场景的名称、时间戳、地图名称等静态信息  
        ego_pose.json：                 （脚本）提供全局车辆的位姿，用于将图像投影
        instance.json                   （脚本）每个实例的名称及其ID
        log.json:                       （脚本）用于scene映射到地图名称
        attribute.json                  （脚本）每个属性的名称及其ID
        visibility.json                 （脚本）每个能见度的名称及其ID
        map.json                        （脚本）每个地图的名称及其文件路径
```
1. 首先根据上文手动自定义calibrated_sensor.json、sensor.json、category.json、scene.json
2. 与上述文件在同一个根目录下执行transform.py转换脚本
3. 执行完成后，会在根目录下生成完整的nuScenes格式的json文件