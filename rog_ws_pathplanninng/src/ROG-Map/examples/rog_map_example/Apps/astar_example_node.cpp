#include "rog_astar/rog_astar.hpp" 
#include <nav_msgs/Odometry.h>
#include <XmlRpcValue.h>  // 添加XmlRpc支持
#include <mutex>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/Point.h>

// 全局变量声明
rog_astar::AStar::Ptr rog_astar_ptr;  // 全局声明
ros::Publisher mkr_pub;
ros::Publisher click_point_pub;  // 添加点击点发布者
std::vector<rog_map::Vec3f> waypoints;
int current_waypoint_group = 0;  // 当前处理的标定点组
bool is_planning = false;
double arrival_threshold = 1.0;  // 修改为1.0米
std::mutex planning_mutex;  // 添加互斥锁
std::vector<bool> reached_waypoints;  // 记录每个标定点是否到达

// 函数提前声明
void publishPointWithText(const rog_map::Vec3f& p,
                          const std::string& text,
                          const rog_map::Color c = rog_map::Color::Green());
void rvizClickCallback(const geometry_msgs::PoseStampedConstPtr& msg);

void odomCallback(const nav_msgs::Odometry::ConstPtr& msg) {
    // 获取当前位置
    Eigen::Vector3d current_pos(msg->pose.pose.position.x,
                               msg->pose.pose.position.y,
                               msg->pose.pose.position.z);

    // 获取标定点
    std::vector<rog_map::Vec3f> waypoints = rog_astar_ptr->getWaypoints();
    if (waypoints.size() % 2 != 0) {
        ROS_ERROR("Invalid number of waypoints. Expected even number, got %zu", waypoints.size());
        return;
    }

    // 打印标定点位置和当前位置
    ROS_INFO_THROTTLE(5.0, "Current waypoint group: %d", current_waypoint_group);
    for (size_t i = 0; i < waypoints.size(); i += 2) {
        ROS_INFO_THROTTLE(5.0, "Group %zu:", i/2);
        ROS_INFO_THROTTLE(5.0, "  Start: [%.3f, %.3f, %.3f]", 
                         waypoints[i](0), waypoints[i](1), waypoints[i](2));
        ROS_INFO_THROTTLE(5.0, "  Goal: [%.3f, %.3f, %.3f]", 
                         waypoints[i+1](0), waypoints[i+1](1), waypoints[i+1](2));
    }
    ROS_INFO_THROTTLE(5.0, "Current position: [%.3f, %.3f, %.3f]", 
                     current_pos(0), current_pos(1), current_pos(2));

    // 检查是否在当前组的起点附近
    if (current_waypoint_group < waypoints.size()/2) {
        int start_idx = current_waypoint_group * 2;
        double distance_to_start = (current_pos - waypoints[start_idx]).norm();
        ROS_INFO_THROTTLE(1.0, "Distance to current start waypoint: %.3f meters (threshold: %.3f)", 
                         distance_to_start, arrival_threshold);
        
        if (!reached_waypoints[current_waypoint_group] && distance_to_start < arrival_threshold) {
            ROS_INFO("=========================================");
            ROS_INFO("Robot has reached waypoint group %d start point!", current_waypoint_group);
            ROS_INFO("Current position: [%.3f, %.3f, %.3f]", 
                    current_pos(0), current_pos(1), current_pos(2));
            ROS_INFO("Start waypoint: [%.3f, %.3f, %.3f]", 
                    waypoints[start_idx](0), waypoints[start_idx](1), waypoints[start_idx](2));
            ROS_INFO("Distance: %.3f meters", distance_to_start);
            ROS_INFO("=========================================");
            
            reached_waypoints[current_waypoint_group] = true;
            
            // 使用互斥锁保护规划过程
            std::lock_guard<std::mutex> lock(planning_mutex);
            
            if (is_planning) {
                ROS_WARN("Already planning, skipping this update");
                return;
            }
            is_planning = true;

            // 执行路径规划（从当前组起点到终点）
            int goal_idx = start_idx + 1;
            ROS_INFO("Starting path planning from waypoint %d to %d", start_idx, goal_idx);
            bool success = rog_astar_ptr->pathSearch(waypoints[start_idx], waypoints[goal_idx], 1.0, 
                                                   rog_astar::ON_INF_MAP | rog_astar::UNKNOWN_AS_FREE);
            
            if (success) {
                ROS_INFO("Path planning successful!");
                // 获取并发布路径
                rog_map::vec_Vec3f path = rog_astar_ptr->getPath();
                ROS_INFO("Path length: %zu points", path.size());
                
                // 发布路径可视化
                visualization_msgs::MarkerArray marker_array;
                
                // 添加路径标记
                visualization_msgs::Marker path_marker;
                path_marker.header.frame_id = "camera_init";
                path_marker.header.stamp = ros::Time::now();
                path_marker.ns = "waypoint_path";
                path_marker.id = current_waypoint_group;
                path_marker.type = visualization_msgs::Marker::LINE_STRIP;
                path_marker.action = visualization_msgs::Marker::ADD;
                path_marker.pose.orientation.x = 0.0;
                path_marker.pose.orientation.y = 0.0;
                path_marker.pose.orientation.z = 0.0;
                path_marker.pose.orientation.w = 1.0;
                path_marker.scale.x = 0.1;
                path_marker.color.r = 1.0;
                path_marker.color.g = 0.0;
                path_marker.color.b = 0.0;
                path_marker.color.a = 1.0;

                for (const auto& point : path) {
                    geometry_msgs::Point p;
                    p.x = point(0);
                    p.y = point(1);
                    p.z = point(2);
                    path_marker.points.push_back(p);
                }
                marker_array.markers.push_back(path_marker);

                // 添加起点和终点标记
                visualization_msgs::Marker start_marker;
                start_marker.header.frame_id = "camera_init";
                start_marker.header.stamp = ros::Time::now();
                start_marker.ns = "path_points";
                start_marker.id = start_idx;
                start_marker.type = visualization_msgs::Marker::SPHERE;
                start_marker.action = visualization_msgs::Marker::ADD;
                start_marker.pose.position.x = waypoints[start_idx](0);
                start_marker.pose.position.y = waypoints[start_idx](1);
                start_marker.pose.position.z = waypoints[start_idx](2);
                start_marker.pose.orientation.x = 0.0;
                start_marker.pose.orientation.y = 0.0;
                start_marker.pose.orientation.z = 0.0;
                start_marker.pose.orientation.w = 1.0;
                start_marker.scale.x = 0.3;
                start_marker.scale.y = 0.3;
                start_marker.scale.z = 0.3;
                start_marker.color.r = 1.0;
                start_marker.color.g = 0.5;
                start_marker.color.b = 0.0;
                start_marker.color.a = 1.0;
                marker_array.markers.push_back(start_marker);

                visualization_msgs::Marker goal_marker;
                goal_marker.header.frame_id = "camera_init";
                goal_marker.header.stamp = ros::Time::now();
                goal_marker.ns = "path_points";
                goal_marker.id = goal_idx;
                goal_marker.type = visualization_msgs::Marker::SPHERE;
                goal_marker.action = visualization_msgs::Marker::ADD;
                goal_marker.pose.position.x = waypoints[goal_idx](0);
                goal_marker.pose.position.y = waypoints[goal_idx](1);
                goal_marker.pose.position.z = waypoints[goal_idx](2);
                goal_marker.pose.orientation.x = 0.0;
                goal_marker.pose.orientation.y = 0.0;
                goal_marker.pose.orientation.z = 0.0;
                goal_marker.pose.orientation.w = 1.0;
                goal_marker.scale.x = 0.3;
                goal_marker.scale.y = 0.3;
                goal_marker.scale.z = 0.3;
                goal_marker.color.r = 0.0;
                goal_marker.color.g = 1.0;
                goal_marker.color.b = 0.0;
                goal_marker.color.a = 1.0;
                marker_array.markers.push_back(goal_marker);

                mkr_pub.publish(marker_array);
                ROS_INFO("Published path markers for group %d", current_waypoint_group);
                
                // 移动到下一组
                current_waypoint_group++;
            } else {
                ROS_ERROR("Path planning failed for group %d!", current_waypoint_group);
            }
            is_planning = false;
        }
    }
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "rm_node");
    ros::NodeHandle nh("~");

    pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);

    /* 1. 创建ROGMap指针 */
    rog_map::ROGMap::Ptr rog_map_ptr = std::make_shared<rog_map::ROGMap>(nh);

    /* 2. 创建路径搜索模块 */
    rog_astar_ptr = std::make_shared<rog_astar::AStar>(nh, rog_map_ptr);

    // 加载到达阈值
    nh.param("/astar/arrival_threshold", arrival_threshold, 1.0);  // 默认值改为1.0
    ROS_INFO("Arrival threshold set to: %f meters", arrival_threshold);

    // 获取标定点
    waypoints = rog_astar_ptr->getWaypoints();
    if (waypoints.size() % 2 != 0) {
        ROS_ERROR("Invalid number of waypoints. Expected even number, got %zu", waypoints.size());
        return -1;
    }
    
    // 初始化到达标记
    reached_waypoints.resize(waypoints.size()/2, false);
    current_waypoint_group = 0;

    // 创建发布者
    mkr_pub = nh.advertise<visualization_msgs::MarkerArray>("/waypoint_markers", 1, true);  // 添加latch=true
    click_point_pub = nh.advertise<visualization_msgs::MarkerArray>("/clicked_points", 1);
    ROS_INFO("Publishing markers to /waypoint_markers and /clicked_points topics");

    // 等待发布者建立连接
    ros::Duration(1.0).sleep();

    // 发布所有标定点标记
    visualization_msgs::MarkerArray initial_markers;
    
    for (size_t i = 0; i < waypoints.size(); i++) {
        visualization_msgs::Marker marker;
        marker.header.frame_id = "camera_init";
        marker.header.stamp = ros::Time::now();
        marker.ns = "waypoints";
        marker.id = i;
        marker.type = visualization_msgs::Marker::SPHERE;
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.position.x = waypoints[i](0);
        marker.pose.position.y = waypoints[i](1);
        marker.pose.position.z = waypoints[i](2);
        marker.pose.orientation.x = 0.0;
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = 0.0;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.3;
        marker.scale.y = 0.3;
        marker.scale.z = 0.3;
        
        // 设置颜色：起点为橙色，终点为绿色
        if (i % 2 == 0) {  // 起点
            marker.color.r = 1.0;
            marker.color.g = 0.5;
            marker.color.b = 0.0;
        } else {  // 终点
            marker.color.r = 0.0;
            marker.color.g = 1.0;
            marker.color.b = 0.0;
        }
        marker.color.a = 1.0;
        marker.lifetime = ros::Duration();
        initial_markers.markers.push_back(marker);
    }

    // 发布初始标记
    mkr_pub.publish(initial_markers);
    ROS_INFO("Published %zu waypoint markers", waypoints.size());
    for (size_t i = 0; i < waypoints.size(); i++) {
        ROS_INFO("Waypoint %zu: [%.3f, %.3f, %.3f]", 
                 i, waypoints[i](0), waypoints[i](1), waypoints[i](2));
    }

    // 创建订阅者
    ros::Subscriber odom_sub = nh.subscribe("/Odometry", 1, odomCallback);
    ros::Subscriber click_sub = nh.subscribe("/clicked_point", 1, rvizClickCallback);
    ROS_INFO("Subscribed to /clicked_point topic for point selection");

    // 定期重新发布标记以确保显示
    ros::Timer timer = nh.createTimer(ros::Duration(1.0), [&](const ros::TimerEvent&) {
        mkr_pub.publish(initial_markers);
    });

    ros::spin();
    return 0;
}

void publishPointWithText(const rog_map::Vec3f& p, const std::string& text, const rog_map::Color c) {
    visualization_msgs::Marker point_marker;
    point_marker.header.frame_id = "world";
    point_marker.header.stamp = ros::Time::now();
    point_marker.ns = text + "_pos";
    point_marker.id = 0;
    point_marker.type = visualization_msgs::Marker::SPHERE;
    point_marker.action = visualization_msgs::Marker::ADD;
    point_marker.pose.position.x = p(0);
    point_marker.pose.position.y = p(1);
    point_marker.pose.position.z = p(2);
    point_marker.pose.orientation.w = 1.0;
    point_marker.scale.x = 0.2;
    point_marker.scale.y = 0.2;
    point_marker.scale.z = 0.2;
    point_marker.color = c;
    point_marker.color.a = 1.0;

    visualization_msgs::Marker text_marker;
    text_marker.header.frame_id = "world";
    text_marker.header.stamp = ros::Time::now();
    text_marker.ns = text;
    text_marker.id = 1;
    text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    text_marker.action = visualization_msgs::Marker::ADD;
    text_marker.pose.position.x = p(0);
    text_marker.pose.position.y = p(1);
    text_marker.pose.position.z = p(2) + 0.3;
    text_marker.pose.orientation.w = 1.0;
    text_marker.scale.z = 0.5;
    text_marker.color = c;
    text_marker.color.a = 1.0;
    text_marker.text = text;

    visualization_msgs::MarkerArray marker_array;
    marker_array.markers.push_back(point_marker);
    marker_array.markers.push_back(text_marker);
    mkr_pub.publish(marker_array);
}

void rvizClickCallback(const geometry_msgs::PoseStampedConstPtr& msg) {
    // 打印点击点的详细信息
    ROS_INFO("=========================================");
    ROS_INFO("Received click point:");
    ROS_INFO("Frame ID: %s", msg->header.frame_id.c_str());
    ROS_INFO("Position: x=%.3f, y=%.3f, z=%.3f", 
             msg->pose.position.x, 
             msg->pose.position.y, 
             msg->pose.position.z);
    ROS_INFO("Orientation: x=%.3f, y=%.3f, z=%.3f, w=%.3f",
             msg->pose.orientation.x,
             msg->pose.orientation.y,
             msg->pose.orientation.z,
             msg->pose.orientation.w);
    ROS_INFO("=========================================");
    
    static rog_map::Vec3f start_pos, goal_pos;
    static bool is_start = true;

    // 设置点击点的高度为0.7
    if (is_start) {
        start_pos = rog_map::Vec3f(msg->pose.position.x, msg->pose.position.y, 0.7);
        is_start = false;
        ROS_INFO("Setting START point at: [%.3f, %.3f, %.3f]", 
                start_pos(0), start_pos(1), start_pos(2));
        publishPointWithText(start_pos, "start", rog_map::Color::Orange());
    }
    else {
        goal_pos = rog_map::Vec3f(msg->pose.position.x, msg->pose.position.y, 0.7);
        ROS_INFO("Setting GOAL point at: [%.3f, %.3f, %.3f]", 
                goal_pos(0), goal_pos(1), goal_pos(2));

        // 发布目标点
        publishPointWithText(goal_pos, "goal", rog_map::Color::Green());
        
        // 执行路径规划
        int flag = rog_astar::UNKNOWN_AS_FREE | rog_astar::ON_INF_MAP;
        bool ret = rog_astar_ptr->pathSearch(start_pos, goal_pos, 0.1, flag);
        if (ret) {
            ROS_INFO("Path found from start to goal");
            // 获取并发布路径
            rog_map::vec_Vec3f path = rog_astar_ptr->getPath();
            visualization_msgs::MarkerArray marker_array;
            visualization_msgs::Marker path_marker;
            path_marker.header.frame_id = "map";
            path_marker.header.stamp = ros::Time::now();
            path_marker.ns = "clicked_path";
            path_marker.id = 0;
            path_marker.type = visualization_msgs::Marker::LINE_STRIP;
            path_marker.action = visualization_msgs::Marker::ADD;
            path_marker.scale.x = 0.1;
            path_marker.color.r = 0.0;
            path_marker.color.g = 1.0;
            path_marker.color.b = 0.0;
            path_marker.color.a = 1.0;

            for (const auto& point : path) {
                geometry_msgs::Point p;
                p.x = point(0);
                p.y = point(1);
                p.z = point(2);
                path_marker.points.push_back(p);
            }

            marker_array.markers.push_back(path_marker);
            mkr_pub.publish(marker_array);
        } else {
            ROS_ERROR("Path not found from start to goal");
        }

        is_start = true;
    }
}
