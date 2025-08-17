/**
 * @file center_line_config.hpp
 * @brief 葡萄藤行间中心线提取器配置
 */

#pragma once

#include <string>
#include <Eigen/Core>

namespace center_line {

struct CenterLineConfig {
    // 地图相关参数
    double resolution = 0.1;  // 地图分辨率(米/格)
    double x_size = 20.0;     // 地图X轴大小(米)
    double y_size = 20.0;     // 地图Y轴大小(米)
    double z_size = 5.0;      // 地图Z轴大小(米)
    
    // 中心线提取参数
    double update_distance = 5.0;             // 更新中心线的行进距离阈值(米)
    double centerline_segment_length = 10.0;  // 每次生成的中心线长度(米)
    double max_corridor_width = 3.0;          // 最大走廊宽度(米)
    double min_confidence_threshold = 0.6;    // 可信度阈值(0-1)
    double step_size = 0.2;                   // 中心线点间距(米)
    
    // 起点位置与方向
    Eigen::Vector3d start_point = Eigen::Vector3d(0.0, 0.0, 0.0);  // 起始点
    Eigen::Vector3d initial_direction = Eigen::Vector3d(1.0, 0.0, 0.0);  // 初始方向
    
    // 话题名称
    std::string odom_topic = "/Odometry";
    std::string pointcloud_topic = "/cloud_registered";
    
    // 坐标系
    std::string map_frame = "map";
    std::string robot_frame = "base_link";
};

}  // namespace center_line
