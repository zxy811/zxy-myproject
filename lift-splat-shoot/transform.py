import json
import os
import glob
import math
from collections import defaultdict

def load_json(path):
    with open(path, "r", encoding='utf-8') as f:
        return json.load(f)

def save_json(path, data):
    # 确保目录存在
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def euler_to_quaternion(roll, pitch, yaw):
    """将欧拉角转换为四元数 (w, x, y, z)"""
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return [w, x, y, z]

def extract_timestamp_from_filename(filename):
    """从文件名提取时间戳"""
    try:
        parts = filename.split("__")
        timestamp_part = parts[-1].split(".")[0]
        return int(timestamp_part)
    except:
        return None

class TrulyUniversal_LSS_NuScenesGenerator:
    def __init__(self, base_path, annotation_path, alias_config_path=None):
        """
        真正通用的LSS兼容nuScenes数据生成器

        Args:
            base_path: 数据根目录
            annotation_path: 标注文件路径
            alias_config_path: 可选的别名配置文件路径，用于智能类别映射
        """
        self.base_path = base_path
        self.annotation_path = annotation_path
        self.alias_config_path = alias_config_path

        # 确保必要的目录存在
        self.ensure_directories()

        # 读取类别配置
        self.load_existing_categories()

        # 加载别名映射
        self.load_alias_config()

        # 路径配置
        self.lidar_path = os.path.join(base_path, "samples/LIDAR_TOP")
        self.camera_paths = {
            'CAM_FRONT_LEFT': os.path.join(base_path, "samples/CAM_FRONT_LEFT"),
            'CAM_FRONT_RIGHT': os.path.join(base_path, "samples/CAM_FRONT_RIGHT"),
            'CAM_BACK_LEFT': os.path.join(base_path, "samples/CAM_BACK_LEFT"),
            'CAM_BACK_RIGHT': os.path.join(base_path, "samples/CAM_BACK_RIGHT")
        }

        # 传感器配置
        self.load_sensor_configs()

        print(f"🎯 从category.json读取到的类别: {list(self.categories_by_name.keys())}")
        print(f"📋 传感器映射: {self.sensor_map}")

    def ensure_directories(self):
        """确保必要的目录存在"""
        dirs_to_create = [
            os.path.join(self.base_path, "v1.0-mini"),
            os.path.join(self.base_path, "maps"),
            os.path.join(self.base_path, "samples/LIDAR_TOP"),
            os.path.join(self.base_path, "samples/CAM_FRONT_LEFT"),
            os.path.join(self.base_path, "samples/CAM_FRONT_RIGHT"),
            os.path.join(self.base_path, "samples/CAM_BACK_LEFT"),
            os.path.join(self.base_path, "samples/CAM_BACK_RIGHT"),
        ]
        
        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)
        
        print(f"✅ 已确保所有必要目录存在")

    def load_existing_categories(self):
        category_file = os.path.join(self.base_path, "v1.0-mini/category.json")
        try:
            self.categories = load_json(category_file)
            self.categories_by_name = {cat['name']: cat for cat in self.categories}
            self.categories_by_token = {cat['token']: cat for cat in self.categories}
            self.category_tokens = {cat['name']: cat['token'] for cat in self.categories}
            self.category_counters = {cat['name']: 1 for cat in self.categories}
        except FileNotFoundError:
            self.create_basic_categories()
        except Exception as e:
            print(f"❌ 读取category.json失败: {e}")
            self.create_basic_categories()

    def create_basic_categories(self):
        basic_categories = [
            {"token": "category_car", "name": "vehicle", "description": "机动车辆"},
            {"token": "category_human", "name": "human", "description": "行人"}
        ]
        category_file = os.path.join(self.base_path, "v1.0-mini/category.json")
        save_json(category_file, basic_categories)
        self.categories = basic_categories
        self.categories_by_name = {cat['name']: cat for cat in basic_categories}
        self.categories_by_token = {cat['token']: cat for cat in basic_categories}
        self.category_tokens = {cat['name']: cat['token'] for cat in basic_categories}
        self.category_counters = {cat['name']: 1 for cat in basic_categories}
        print("✅ 创建了基础的category.json文件")

    def load_alias_config(self):
        self.alias_mapping = {}
        if self.alias_config_path and os.path.exists(self.alias_config_path):
            try:
                alias_config = load_json(self.alias_config_path)
                for standard_category, aliases in alias_config.items():
                    if standard_category in self.categories_by_name:
                        self.alias_mapping[standard_category.lower()] = standard_category
                        for alias in aliases:
                            self.alias_mapping[alias.lower()] = standard_category
                print(f"✅ 加载别名配置，支持 {len(self.alias_mapping)} 个别名映射")
            except Exception as e:
                print(f"⚠️ 加载别名配置失败: {e}")
        else:
            self.create_basic_alias_mapping()

    def create_basic_alias_mapping(self):
        basic_aliases = {
            'vehicle': ['car', 'auto', 'automobile', 'truck', 'bus', 'van', 'taxi'],
            'human': ['person', 'people', 'pedestrian', 'man', 'woman', 'child', 'adult', 'walker'],
            'bicycle': ['bike', 'cycle', 'motorcycle', 'motorbike', 'scooter'],
            'barrier': ['cone', 'fence', 'wall', 'obstacle'],
            'sign': ['traffic_sign', 'road_sign', 'billboard', 'signal']
        }
        self.alias_mapping = {}
        for category_name in self.categories_by_name.keys():
            self.alias_mapping[category_name.lower()] = category_name
            if category_name in basic_aliases:
                for alias in basic_aliases[category_name]:
                    self.alias_mapping[alias.lower()] = category_name

    def load_sensor_configs(self):
        try:
            self.sensors = load_json(os.path.join(self.base_path, "v1.0-mini/sensor.json"))
            self.calibrated_sensors = load_json(os.path.join(self.base_path, "v1.0-mini/calibrated_sensor.json"))
            self.sensor_map = {s['channel']: s['token'] for s in self.sensors}
            self.calib_sensor_map = {}
            for cs in self.calibrated_sensors:
                for s in self.sensors:
                    if s['token'] == cs['sensor_token']:
                        self.calib_sensor_map[s['channel']] = cs['token']
                        break
        except FileNotFoundError:
            print("⚠️ 传感器配置文件不存在，将创建默认配置")
            self.create_default_sensor_configs()

    def create_default_sensor_configs(self):
        """创建默认的传感器配置"""
        # 创建sensor.json
        sensors = [
            {"token": "sensor_cam_front_left", "channel": "CAM_FRONT_LEFT", "modality": "camera"},
            {"token": "sensor_cam_front_right", "channel": "CAM_FRONT_RIGHT", "modality": "camera"},
            {"token": "sensor_cam_back_left", "channel": "CAM_BACK_LEFT", "modality": "camera"},
            {"token": "sensor_cam_back_right", "channel": "CAM_BACK_RIGHT", "modality": "camera"},
            {"token": "sensor_lidar_top", "channel": "LIDAR_TOP", "modality": "lidar"}
        ]
        save_json(os.path.join(self.base_path, "v1.0-mini/sensor.json"), sensors)

        # 创建calibrated_sensor.json
        calibrated_sensors = []
        for sensor in sensors:
            if sensor['modality'] == 'camera':
                calib = {
                    "token": f"calib_{sensor['token']}",
                    "sensor_token": sensor['token'],
                    "translation": [0.0, 0.0, 1.5],
                    "rotation": [1.0, 0.0, 0.0, 0.0],
                    "camera_intrinsic": [
                        [1600.0, 0.0, 800.0],
                        [0.0, 900.0, 450.0],
                        [0.0, 0.0, 1.0]
                    ]
                }
            else:  # lidar
                calib = {
                    "token": f"calib_{sensor['token']}",
                    "sensor_token": sensor['token'],
                    "translation": [0.0, 0.0, 1.8],
                    "rotation": [1.0, 0.0, 0.0, 0.0],
                    "camera_intrinsic": []
                }
            calibrated_sensors.append(calib)
        
        save_json(os.path.join(self.base_path, "v1.0-mini/calibrated_sensor.json"), calibrated_sensors)
        
        # 重新加载配置
        self.sensors = sensors
        self.calibrated_sensors = calibrated_sensors
        self.sensor_map = {s['channel']: s['token'] for s in self.sensors}
        self.calib_sensor_map = {}
        for cs in self.calibrated_sensors:
            for s in self.sensors:
                if s['token'] == cs['sensor_token']:
                    self.calib_sensor_map[s['channel']] = cs['token']
                    break
        
        print("✅ 创建了默认的传感器配置文件")

    def create_dummy_map_file(self):
        """创建虚拟地图文件以满足NuScenes库的要求"""
        maps_dir = os.path.join(self.base_path, "maps")
        os.makedirs(maps_dir, exist_ok=True)
        
        # 创建一个简单的PNG地图文件（100x100像素的空白图片）
        map_file_path = os.path.join(maps_dir, "custom_map.png")
        if not os.path.exists(map_file_path):
            try:
                from PIL import Image
                import numpy as np
                # 创建一个100x100的空白图片
                img_array = np.zeros((100, 100, 4), dtype=np.uint8)  # RGBA
                img = Image.fromarray(img_array, 'RGBA')
                img.save(map_file_path)
                print(f"✅ 创建虚拟地图文件: {map_file_path}")
            except ImportError:
                # 如果没有PIL，创建一个空文件
                with open(map_file_path, 'wb') as f:
                    # 写入一个最小的PNG文件头
                    png_header = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00d\x00\x00\x00d\x08\x02\x00\x00\x00\xff\x80\xbb\x07'
                    f.write(png_header)
                print(f"✅ 创建空地图文件: {map_file_path}")

    def generate_default_metadata_files(self):
        meta_path = os.path.join(self.base_path, "v1.0-mini")
        os.makedirs(meta_path, exist_ok=True)

        log_file = os.path.join(meta_path, "log.json")
        if not os.path.exists(log_file):
            logs = [{
                "token": "log_custom_001",
                "logfile": "custom_logfile",
                "vehicle": "vehicle_001",
                "date_captured": "2025-01-01",
                "location": "custom-dataset"
            }]
            save_json(log_file, logs)
            print("✅ 生成 log.json (默认配置)")

        vis_file = os.path.join(meta_path, "visibility.json")
        if not os.path.exists(vis_file):
            visibility = [
                {"description": "0-40% visible", "token": "vis_1", "level": "v0-40"},
                {"description": "40-60% visible", "token": "vis_2", "level": "v40-60"},
                {"description": "60-80% visible", "token": "vis_3", "level": "v60-80"},
                {"description": "80-100% visible", "token": "vis_4", "level": "v80-100"}
            ]
            save_json(vis_file, visibility)
            print("✅ 生成 visibility.json (默认配置)")

        attr_file = os.path.join(meta_path, "attribute.json")
        if not os.path.exists(attr_file):
            attributes = [
                {"token": "attr_vehicle_moving", "name": "vehicle.moving", "description": "Vehicle is moving."},
                {"token": "attr_vehicle_stopped", "name": "vehicle.stopped", "description": "Vehicle is stationary."},
                {"token": "attr_pedestrian_standing", "name": "pedestrian.standing", "description": "Human is standing."}
            ]
            save_json(attr_file, attributes)
            print("✅ 生成 attribute.json (默认配置)")

        # 修改：使用PNG格式而不是JSON
        map_file = os.path.join(meta_path, "map.json")
        if not os.path.exists(map_file):
            self.create_dummy_map_file()  # 先创建地图文件
            maps = [{
                "token": "map_custom_001",
                "log_tokens": ["log_custom_001"],
                "category": "semantic_prior",
                "filename": "maps/custom_map.png"  # 修改：使用PNG格式
            }]
            save_json(map_file, maps)
            print("✅ 生成 map.json (默认配置)")

    def generate_scene_json(self):
        """生成scene.json文件"""
        lidar_count = len(glob.glob(os.path.join(self.lidar_path, "*.pcd.bin")))
        scenes = [{
            "token": "orchard_scene_001",
            "log_token": "log_custom_001",
            "nbr_samples": lidar_count,
            "first_sample_token": "sample_000001",
            "last_sample_token": f"sample_{lidar_count:06d}",
            "name": "custom_scene_001",
            "description": "Generated scene from custom dataset"
        }]
        
        output_path = os.path.join(self.base_path, "v1.0-mini/scene.json")
        save_json(output_path, scenes)
        print(f"✅ scene.json 生成完成：共 {len(scenes)} 个场景")
        return scenes

    def smart_normalize_category(self, raw_category):
        if not raw_category:
            return list(self.categories_by_name.keys())[0] if self.categories_by_name else 'unknown'
        raw_category_lower = raw_category.lower().strip()
        for category_name in self.categories_by_name.keys():
            if raw_category_lower == category_name.lower():
                return category_name
        if raw_category_lower in self.alias_mapping:
            return self.alias_mapping[raw_category_lower]
        for category_name in self.categories_by_name.keys():
            if raw_category_lower in category_name.lower() or category_name.lower() in raw_category_lower:
                return category_name
        return self.handle_unknown_category(raw_category)

    def handle_unknown_category(self, raw_category):
        available_categories = list(self.categories_by_name.keys())
        if 'object' in available_categories:
            return 'object'
        return available_categories[0] if available_categories else 'vehicle'

    def analyze_annotation_categories(self):
        print("\n🔍 分析标注文件中的类别分布...")
        ann_files = glob.glob(os.path.join(self.annotation_path, "*.pcd.pcd.json"))
        if not ann_files:
            print("⚠️ 未找到标注文件，将跳过类别分析")
            return {}, {}
            
        raw_categories = defaultdict(int)
        mapping_preview = {}
        for ann_file in ann_files[:3]:
            try:
                ann_data = load_json(ann_file)
                figures = ann_data.get("figures", [])
                objects = ann_data.get("objects", [])
                for fig in figures:
                    obj_key = fig["objectKey"]
                    obj = next((o for o in objects if o["key"] == obj_key), None)
                    if obj:
                        raw_category = obj.get("classTitle", "")
                        if raw_category:
                            raw_categories[raw_category] += 1
                            if raw_category not in mapping_preview:
                                normalized = self.smart_normalize_category(raw_category)
                                mapping_preview[raw_category] = normalized
            except Exception as e:
                print(f"⚠️ 分析文件失败 {ann_file}: {e}")
        print(f"\n📊 发现的原始类别及其映射预览:")
        for raw_cat, count in raw_categories.items():
            mapped_cat = mapping_preview.get(raw_cat, "未映射")
            print(f"   - '{raw_cat}' ({count}次) → '{mapped_cat}'")
        return raw_categories, mapping_preview

    def generate_sample_json(self):
        lidar_files = sorted(glob.glob(os.path.join(self.lidar_path, "*.pcd.bin")))
        if not lidar_files:
            print("⚠️ 未找到LIDAR文件，将创建空的sample.json")
            save_json(os.path.join(self.base_path, "v1.0-mini/sample.json"), [])
            return []
            
        samples = []
        
        for i, lidar_file in enumerate(lidar_files):
            filename = os.path.basename(lidar_file)
            timestamp = extract_timestamp_from_filename(filename)
            if timestamp is None:
                timestamp = 1000000 + i * 100000  # 生成假时间戳
                print(f"警告：无法提取时间戳: {filename}，使用假时间戳: {timestamp}")
            
            sample_token = f"sample_{i+1:06d}"
            
            # 修改：添加data字段，这是LSS代码必需的
            sample_data = {}
            
            # 添加LIDAR数据
            sample_data['LIDAR_TOP'] = f"data_lidar_top_{sample_token}"
            
            # 添加相机数据
            for camera_name in self.camera_paths.keys():
                camera_name_lower = camera_name.lower()
                sample_data[camera_name] = f"data_{camera_name_lower}_{sample_token}"
            
            sample = {
                "token": sample_token,
                "timestamp": timestamp,
                "scene_token": "orchard_scene_001",
                "prev": f"sample_{i:06d}" if i > 0 else "",
                "next": f"sample_{i+2:06d}" if i < len(lidar_files)-1 else "",
                "data": sample_data,  # 新增：必需的data字段
                "anns": []  # 稍后会填充标注token
            }
            samples.append(sample)
        
        output_path = os.path.join(self.base_path, "v1.0-mini/sample.json")
        save_json(output_path, samples)
        print(f"✅ sample.json 生成完成：共 {len(samples)} 帧")
        return samples

    def find_file_by_timestamp(self, folder_path, target_timestamp, pattern):
        if not os.path.exists(folder_path):
            return None
        files = glob.glob(os.path.join(folder_path, pattern))
        for file_path in files:
            filename = os.path.basename(file_path)
            file_timestamp = extract_timestamp_from_filename(filename)
            if file_timestamp and abs(file_timestamp - target_timestamp) < 100000:
                return filename
        return None

    def generate_sample_data_json(self, samples):
        sample_data_list = []
        for sample in samples:
            sample_token = sample['token']
            timestamp = sample['timestamp']
            sample_num = sample_token.split('_')[-1]
            
            # LIDAR数据
            lidar_filename = self.find_file_by_timestamp(self.lidar_path, timestamp, "*.pcd.bin")
            if lidar_filename:
                lidar_data = {
                    "token": f"data_lidar_top_{sample_token}",
                    "sample_token": sample_token,
                    "ego_pose_token": sample_token,
                    "calibrated_sensor_token": self.calib_sensor_map['LIDAR_TOP'],
                    "timestamp": timestamp,
                    "fileformat": "pcd",
                    "is_key_frame": True,
                    "height": 0,
                    "width": 0,
                    "filename": f"samples/LIDAR_TOP/{lidar_filename}",
                    "prev": f"data_lidar_top_sample_{int(sample_num)-1:06d}" if int(sample_num) > 1 else "",
                    "next": f"data_lidar_top_sample_{int(sample_num)+1:06d}" if int(sample_num) < len(samples) else ""
                }
                sample_data_list.append(lidar_data)
            
            # 相机数据
            for camera_name, camera_path in self.camera_paths.items():
                camera_filename = self.find_file_by_timestamp(camera_path, timestamp, "*.jpg")
                if camera_filename:
                    camera_name_lower = camera_name.lower()
                    camera_data = {
                        "token": f"data_{camera_name_lower}_{sample_token}",
                        "sample_token": sample_token,
                        "ego_pose_token": sample_token,
                        "calibrated_sensor_token": self.calib_sensor_map[camera_name],
                        "timestamp": timestamp,
                        "fileformat": "jpg",
                        "is_key_frame": True,
                        "height": 900,
                        "width": 1600,
                        "filename": f"samples/{camera_name}/{camera_filename}",
                        "prev": f"data_{camera_name_lower}_sample_{int(sample_num)-1:06d}" if int(sample_num) > 1 else "",
                        "next": f"data_{camera_name_lower}_sample_{int(sample_num)+1:06d}" if int(sample_num) < len(samples) else ""
                    }
                    sample_data_list.append(camera_data)
        
        output_path = os.path.join(self.base_path, "v1.0-mini/sample_data.json")
        save_json(output_path, sample_data_list)
        print(f"✅ sample_data.json 生成完成：共 {len(sample_data_list)} 条数据")

    def generate_truly_universal_sample_annotation_json(self, samples):
        annotations = []
        instances_info = []
        
        # 更新samples的anns字段
        for sample in samples:
            sample_token = sample['token']
            timestamp = sample['timestamp']
            sample_ann_tokens = []
            
            ann_files = glob.glob(os.path.join(self.annotation_path, "*.pcd.pcd.json"))
            matching_ann_file = None
            for ann_file in ann_files:
                ann_filename = os.path.basename(ann_file)
                ann_timestamp = extract_timestamp_from_filename(ann_filename)
                if ann_timestamp and abs(ann_timestamp - timestamp) < 100000:
                    matching_ann_file = ann_file
                    break
            
            if not matching_ann_file:
                print(f"⚠️ 警告：未找到匹配的标注文件: {sample_token}")
                continue
                
            try:
                ann_data = load_json(matching_ann_file)
            except Exception as e:
                print(f"❌ 错误：无法读取标注文件 {matching_ann_file}: {e}")
                continue
                
            figures = ann_data.get("figures", [])
            objects = ann_data.get("objects", [])
            for fig in figures:
                obj_key = fig["objectKey"]
                obj = next((o for o in objects if o["key"] == obj_key), None)
                if not obj:
                    continue
                    
                raw_category = obj.get("classTitle", "")
                normalized_category = self.smart_normalize_category(raw_category)
                if normalized_category not in self.categories_by_name:
                    continue
                    
                instance_token = f"{normalized_category}_inst_{self.category_counters[normalized_category]:03d}"
                self.category_counters[normalized_category] += 1
                geom = fig["geometry"]
                rotation_info = geom.get("rotation", {"x": 0.0, "y": 0.0, "z": 0.0})
                quaternion = euler_to_quaternion(rotation_info.get("x", 0.0),
                                                 rotation_info.get("y", 0.0),
                                                 rotation_info.get("z", 0.0))
                ann_token = f"ann_{sample_token}_{instance_token}"
                sample_ann_tokens.append(ann_token)  # 添加到sample的anns列表
                
                annotation = {
                    "token": ann_token,
                    "sample_token": sample_token,
                    "instance_token": instance_token,
                    "visibility_token": "vis_4",
                    "attribute_tokens": [],
                    "translation": [
                        geom["position"]["x"],
                        geom["position"]["y"],
                        geom["position"]["z"]
                    ],
                    "size": [
                        geom["dimensions"]["x"],
                        geom["dimensions"]["y"],
                        geom["dimensions"]["z"]
                    ],
                    "rotation": quaternion,
                    "prev": "",
                    "next": "",
                    "num_lidar_pts": 0,
                    "num_radar_pts": 0,
                    "category_name": f"{normalized_category}.default"  # 新增：LSS需要的字段
                }
                annotations.append(annotation)
                instances_info.append({
                    'token': instance_token,
                    'category': normalized_category,
                    'annotation_token': ann_token
                })
            
            sample['anns'] = sample_ann_tokens  # 更新sample的anns字段
        
        # 重新保存更新后的samples
        samples_output_path = os.path.join(self.base_path, "v1.0-mini/sample.json")
        save_json(samples_output_path, samples)
        
        output_path = os.path.join(self.base_path, "v1.0-mini/sample_annotation.json")
        save_json(output_path, annotations)
        print(f"✅ sample_annotation.json 生成完成：共 {len(annotations)} 条标注")
        return annotations, instances_info

    def generate_truly_universal_instance_json(self, instances_info):
        instances = []
        for instance_info in instances_info:
            instance_token = instance_info['token']
            category = instance_info['category']
            annotation_token = instance_info['annotation_token']
            category_token = self.category_tokens[category]
            instance = {
                "token": instance_token,
                "category_token": category_token,
                "nbr_annotations": 1,
                "first_annotation_token": annotation_token,
                "last_annotation_token": annotation_token
            }
            instances.append(instance)
        output_path = os.path.join(self.base_path, "v1.0-mini/instance.json")
        save_json(output_path, instances)
        print(f"✅ instance.json 生成完成：共 {len(instances)} 个实例")

    def generate_simple_ego_pose_json(self, samples):
        ego_poses = []
        for sample in samples:
            sample_token = sample['token']
            timestamp = sample['timestamp']
            ego_pose = {
                "token": sample_token,
                "translation": [0.0, 0.0, 0.0],
                "rotation": [1.0, 0.0, 0.0, 0.0],
                "timestamp": timestamp
            }
            ego_poses.append(ego_pose)
        output_path = os.path.join(self.base_path, "v1.0-mini/ego_pose.json")
        save_json(output_path, ego_poses)
        print(f"✅ ego_pose.json 生成完成：共 {len(ego_poses)} 个位姿")
        return ego_poses

    def validate_lss_compatibility(self):
        required_files = [
            "sample.json", "sample_data.json", "sample_annotation.json",
            "instance.json", "category.json", "calibrated_sensor.json",
            "sensor.json", "scene.json", "ego_pose.json", "map.json"
        ]
        missing_files = []
        for file in required_files:
            file_path = os.path.join(self.base_path, f"v1.0-mini/{file}")
            if not os.path.exists(file_path):
                missing_files.append(file)
        if missing_files:
            print(f"   ❌ 缺少文件: {missing_files}")
            return False
        else:
            print("   ✅ 所有必需文件都存在")
            print("   🎉 数据完全兼容LSS格式！")
            return True

    def generate_all(self):
        print("🚀 开始生成真正通用LSS兼容版nuscenes格式数据...")
        self.generate_default_metadata_files()
        self.generate_scene_json()  # 新增：生成scene.json
        self.analyze_annotation_categories()
        samples = self.generate_sample_json()
        self.generate_sample_data_json(samples)
        annotations, instances_info = self.generate_truly_universal_sample_annotation_json(samples)
        self.generate_truly_universal_instance_json(instances_info)
        self.generate_simple_ego_pose_json(samples)
        self.validate_lss_compatibility()
        print("🎉 数据生成完成！")

if __name__ == "__main__":
    base_path = "/home/zxy/lift-splat-shoot/nuScenes-1/mini"
    annotation_path = "/home/zxy/lift-splat-shoot/nuScenes-1/mini/samples/project_3_annotations_2025_07_24_07_40_23_sly point cloud format 1.0/ds0/ann"
    alias_config_path = None
    generator = TrulyUniversal_LSS_NuScenesGenerator(
        base_path=base_path,
        annotation_path=annotation_path,
        alias_config_path=alias_config_path
    )
    generator.generate_all()
