/**
 * @file center_line_node.cpp
 * @brief 葡萄藤行间中心线提取ROS节点
 */

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/Point.h>
#include <Eigen/Core>

#include "center_line/center_line.hpp"
#include "center_line/center_line_config.hpp"
#include "rog_map/rog_map.h"

class CenterLineNode {
public:
    CenterLineNode(const ros::NodeHandle& nh, const ros::NodeHandle& pnh)
        : nh_(nh), pnh_(pnh) {
        // 加载配置
        loadConfig();
        
        // 初始化ROG-Map（使用正确的构造函数）
        initializeROGMap();
        
        // 初始化中心线提取器
        center_line_ = std::make_shared<center_line::CenterLine>(
            rog_map_, config_.max_corridor_width, config_.min_confidence_threshold);
        
        // 发布器
        centerline_pub_ = pnh_.advertise<visualization_msgs::Marker>("center_line", 1);
        
        // 订阅器
        odom_sub_ = nh_.subscribe(config_.odom_topic, 10, &CenterLineNode::odomCallback, this);
        pointcloud_sub_ = nh_.subscribe(config_.pointcloud_topic, 1, &CenterLineNode::pointcloudCallback, this);
        
        // 定时器
        vis_timer_ = pnh_.createTimer(ros::Duration(0.1), &CenterLineNode::visTimerCallback, this);
        
        // 初始化状态
        current_position_ = config_.start_point;
        current_direction_ = config_.initial_direction.normalized();
        last_update_position_ = current_position_;
        initialized_ = false;
        
        ROS_INFO("Center Line Node initialized successfully!");
    }
    
private:
    // ROS节点句柄
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    
    // 订阅与发布
    ros::Subscriber odom_sub_;
    ros::Subscriber pointcloud_sub_;
    ros::Publisher centerline_pub_;
    ros::Timer vis_timer_;
    
    // 配置
    center_line::CenterLineConfig config_;
    
    // ROG-Map与中心线提取器
    std::shared_ptr<rog_map::ROGMap> rog_map_;
    std::shared_ptr<center_line::CenterLine> center_line_;
    
    // 全局中心线
    std::vector<Eigen::Vector3d> global_centerline_;
    
    // 当前状态
    Eigen::Vector3d current_position_;
    Eigen::Vector3d current_direction_;
    Eigen::Vector3d last_update_position_;
    bool initialized_ = false;
    
    // 加载配置
    void loadConfig() {
        // 地图参数
        pnh_.param<double>("resolution", config_.resolution, config_.resolution);
        pnh_.param<double>("x_size", config_.x_size, config_.x_size);
        pnh_.param<double>("y_size", config_.y_size, config_.y_size);
        pnh_.param<double>("z_size", config_.z_size, config_.z_size);
        
        // 中心线参数
        pnh_.param<double>("update_distance", config_.update_distance, config_.update_distance);
        pnh_.param<double>("centerline_segment_length", config_.centerline_segment_length, config_.centerline_segment_length);
        pnh_.param<double>("max_corridor_width", config_.max_corridor_width, config_.max_corridor_width);
        pnh_.param<double>("min_confidence_threshold", config_.min_confidence_threshold, config_.min_confidence_threshold);
        pnh_.param<double>("step_size", config_.step_size, config_.step_size);
        
        // 起点与方向
        double start_x, start_y, start_z, dir_x, dir_y, dir_z;
        pnh_.param<double>("start_x", start_x, config_.start_point.x());
        pnh_.param<double>("start_y", start_y, config_.start_point.y());
        pnh_.param<double>("start_z", start_z, config_.start_point.z());
        pnh_.param<double>("dir_x", dir_x, config_.initial_direction.x());
        pnh_.param<double>("dir_y", dir_y, config_.initial_direction.y());
        pnh_.param<double>("dir_z", dir_z, config_.initial_direction.z());
        
        config_.start_point = Eigen::Vector3d(start_x, start_y, start_z);
        config_.initial_direction = Eigen::Vector3d(dir_x, dir_y, dir_z).normalized();
        
        // 话题与坐标系
        pnh_.param<std::string>("odom_topic", config_.odom_topic, config_.odom_topic);
        pnh_.param<std::string>("pointcloud_topic", config_.pointcloud_topic, config_.pointcloud_topic);
        pnh_.param<std::string>("map_frame", config_.map_frame, config_.map_frame);
        pnh_.param<std::string>("robot_frame", config_.robot_frame, config_.robot_frame);
        
        ROS_INFO("Center Line Config loaded.");
        ROS_INFO("  - Resolution: %.2f", config_.resolution);
        ROS_INFO("  - Map size: %.1f x %.1f x %.1f", config_.x_size, config_.y_size, config_.z_size);
        ROS_INFO("  - Update distance: %.2f", config_.update_distance);
        ROS_INFO("  - Segment length: %.2f", config_.centerline_segment_length);
        ROS_INFO("  - Max corridor width: %.2f", config_.max_corridor_width);
        ROS_INFO("  - Min confidence: %.2f", config_.min_confidence_threshold);
    }
    
    // 初始化ROG-Map
    void initializeROGMap() {
        try {
            // 使用正确的构造函数初始化ROGMap，传入NodeHandle
            rog_map_ = std::make_shared<rog_map::ROGMap>(nh_);
            
            // 这里不再调用setMapParams方法，因为ROGMap的参数通过ROS参数服务器设置
            ROS_INFO("ROG-Map initialized successfully");
        } catch (const std::exception& e) {
            ROS_ERROR("Failed to initialize ROG-Map: %s", e.what());
            throw;
        }
    }
    
    // 回调：里程计
    void odomCallback(const nav_msgs::Odometry::ConstPtr& msg) {
        try {
            // 更新当前位置
            current_position_ = Eigen::Vector3d(
                msg->pose.pose.position.x,
                msg->pose.pose.position.y,
                msg->pose.pose.position.z
            );
            
            // 计算当前方向
            Eigen::Quaterniond q(
                msg->pose.pose.orientation.w,
                msg->pose.pose.orientation.x,
                msg->pose.pose.orientation.y,
                msg->pose.pose.orientation.z
            );
            Eigen::Vector3d forward_direction = q * Eigen::Vector3d::UnitX();
            current_direction_ = forward_direction.normalized();
            
            // 如果是第一次接收到里程计数据，初始化系统
            if (!initialized_) {
                last_update_position_ = current_position_;
                initialized_ = true;
                ROS_INFO("System initialized with first odometry at position (%.2f, %.2f, %.2f)",
                        current_position_.x(), current_position_.y(), current_position_.z());
                
                // 生成初始中心线
                updateCenterline();
                return;
            }
            
            // 计算距上次更新的距离
            double traveled_distance = (current_position_ - last_update_position_).norm();
            
            // 当行进距离超过阈值时，更新中心线
            if (traveled_distance >= config_.update_distance) {
                ROS_INFO("Traveled %.2f meters, updating centerline", traveled_distance);
                updateCenterline();
                last_update_position_ = current_position_;
            }
        } catch (const std::exception& e) {
            ROS_ERROR("Error in odomCallback: %s", e.what());
        }
    }
    
    // 回调：点云
    void pointcloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg) {
        try {
            // 将ROS PointCloud2消息转换为PCL点云
            rog_map::PointCloud cloud;
            pcl::fromROSMsg(*msg, cloud);
            
            // 创建当前机器人姿态
            rog_map::Pose pose;
            
            // 使用相同类型(float)创建位置向量
            pose.first = rog_map::Vec3f(
                static_cast<float>(current_position_.x()), 
                static_cast<float>(current_position_.y()), 
                static_cast<float>(current_position_.z())
            );
            
            // 创建四元数 - 使用相同类型(float)
            rog_map::Quatf quat(1.0f, 0.0f, 0.0f, 0.0f);
            pose.second = quat;
            
            // 更新地图
            rog_map_->updateMap(cloud, pose);
            
            ROS_DEBUG("Updated ROG-Map with point cloud data");
        } catch (const std::exception& e) {
            ROS_ERROR("Error in pointcloudCallback: %s", e.what());
        }
    }
    
    // 回调：可视化定时器
    void visTimerCallback(const ros::TimerEvent& event) {
        try {
            publishCenterlineVisualization();
        } catch (const std::exception& e) {
            ROS_ERROR("Error in visTimerCallback: %s", e.what());
        }
    }
    
    // 更新中心线
    void updateCenterline() {
        try {
            // 使用中心线提取器生成新的中心线段
            std::vector<Eigen::Vector3d> new_segment = center_line_->extractCenterline(
                current_position_, current_direction_, config_.centerline_segment_length, config_.step_size);
            
            if (new_segment.empty()) {
                ROS_WARN("Failed to extract centerline segment");
                return;
            }
            
            ROS_INFO("Generated centerline segment with %zu points", new_segment.size());
            
            // 将新段与全局中心线合并
            if (global_centerline_.empty()) {
                global_centerline_ = new_segment;
            } else {
                mergeCenterlineSegments(new_segment);
            }
        } catch (const std::exception& e) {
            ROS_ERROR("Error in updateCenterline: %s", e.what());
        }
    }
    
    // 合并中心线段
    void mergeCenterlineSegments(const std::vector<Eigen::Vector3d>& new_segment) {
        // 寻找最佳连接点
        int best_connection_idx = -1;
        double min_distance = std::numeric_limits<double>::max();
        
        // 查找全局中心线中与新段起点最近的点
        for (size_t i = 0; i < global_centerline_.size(); ++i) {
            double dist = (global_centerline_[i] - new_segment[0]).norm();
            if (dist < min_distance) {
                min_distance = dist;
                best_connection_idx = i;
            }
        }
        
        // 如果找到较近的连接点，进行平滑连接
        if (best_connection_idx != -1 && min_distance < 1.0) { // 1米作为连接阈值
            // 保留全局中心线的前部
            std::vector<Eigen::Vector3d> merged_centerline(
                global_centerline_.begin(), 
                global_centerline_.begin() + best_connection_idx + 1
            );
            
            // 添加新段（跳过第一个点以避免重复）
            merged_centerline.insert(
                merged_centerline.end(),
                new_segment.begin() + 1,
                new_segment.end()
            );
            
            global_centerline_ = merged_centerline;
            ROS_INFO("Merged new segment with existing centerline at point %d", best_connection_idx);
        } else {
            // 如果没有找到好的连接点，直接添加到末尾
            global_centerline_.insert(
                global_centerline_.end(),
                new_segment.begin(),
                new_segment.end()
            );
            ROS_INFO("Appended new segment to existing centerline");
        }
    }
    
    // 发布中心线可视化
    void publishCenterlineVisualization() {
        if (global_centerline_.empty()) return;
        
        visualization_msgs::Marker marker;
        marker.header.frame_id = config_.map_frame;
        marker.header.stamp = ros::Time::now();
        marker.ns = "vine_row_centerline";
        marker.id = 0;
        marker.type = visualization_msgs::Marker::LINE_STRIP;
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.1;  // 线宽
        marker.color.r = 0.0;
        marker.color.g = 1.0;
        marker.color.b = 0.0;
        marker.color.a = 1.0;
        marker.lifetime = ros::Duration();
        
        for (const auto& point : global_centerline_) {
            geometry_msgs::Point p;
            p.x = point.x();
            p.y = point.y();
            p.z = point.z();
            marker.points.push_back(p);
        }
        
        centerline_pub_.publish(marker);
    }
};

int main(int argc, char** argv) {
    try {
        ros::init(argc, argv, "center_line_node");
        ros::NodeHandle nh;
        ros::NodeHandle pnh("~");
        
        // 使用异常处理来捕获初始化过程中的错误
        CenterLineNode node(nh, pnh);
        
        ROS_INFO("Center Line Node is running...");
        ros::spin();
        
        return 0;
    } catch (const std::exception& e) {
        ROS_ERROR("Caught exception: %s", e.what());
        return 1;
    }
}
