#ifndef ROG_ASTAR_CONFIG_HPP
#define ROG_ASTAR_CONFIG_HPP

#include <ros/ros.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <type_traits>
#include <vector>
#include <XmlRpcValue.h>

namespace rog_astar {
    using std::string;
    using std::vector;
    using Vec3f = Eigen::Vector3d;
    using Vec3i = Eigen::Vector3i;
    typedef pcl::PointXYZINormal PclPoint;
    typedef pcl::PointCloud<PclPoint> PointCloud;

    class Config {
    private:
        ros::NodeHandle nh_;

        // 处理基本类型
        template<class T>
        typename std::enable_if<!std::is_same<T, std::vector<double>>::value && 
                              !std::is_same<T, std::vector<int>>::value && 
                              !std::is_same<T, std::vector<std::vector<double>>>::value, bool>::type
        LoadParam(string param_name, T &param_value, T default_value = T{}, bool required = false) {
            if (nh_.getParam(param_name, param_value)) {
                printf("\033[0;32m Load param %s success: \033[0;0m", (nh_.getNamespace() + "/" + param_name).c_str());
                std::cout << param_value << std::endl;
                return true;
            } else {
                printf("\033[0;33m Load param %s failed, use default: \033[0;0m",
                       (nh_.getNamespace() + "/" + param_name).c_str());
                param_value = default_value;
                std::cout << param_value << std::endl;
                if (required) throw std::invalid_argument("Required param not found");
                return false;
            }
        }

        // 处理vector<double>类型
        bool LoadParam(string param_name, std::vector<double> &param_value, std::vector<double> default_value = {}, bool required = false) {
            if (nh_.getParam(param_name, param_value)) {
                printf("\033[0;32m Load param %s success: \033[0;0m", (nh_.getNamespace() + "/" + param_name).c_str());
                for (const auto& val : param_value) std::cout << val << " ";
                std::cout << std::endl;
                return true;
            } else {
                printf("\033[0;33m Load param %s failed, use default: \033[0;0m",
                       (nh_.getNamespace() + "/" + param_name).c_str());
                param_value = default_value;
                for (const auto& val : param_value) std::cout << val << " ";
                std::cout << std::endl;
                if (required) throw std::invalid_argument("Required param not found");
                return false;
            }
        }

        // 处理vector<int>类型
        bool LoadParam(string param_name, std::vector<int> &param_value, std::vector<int> default_value = {}, bool required = false) {
            if (nh_.getParam(param_name, param_value)) {
                printf("\033[0;32m Load param %s success: \033[0;0m", (nh_.getNamespace() + "/" + param_name).c_str());
                for (const auto& val : param_value) std::cout << val << " ";
                std::cout << std::endl;
                return true;
            } else {
                printf("\033[0;33m Load param %s failed, use default: \033[0;0m",
                       (nh_.getNamespace() + "/" + param_name).c_str());
                param_value = default_value;
                for (const auto& val : param_value) std::cout << val << " ";
                std::cout << std::endl;
                if (required) throw std::invalid_argument("Required param not found");
                return false;
            }
        }

        // 处理vector<vector<double>>类型
        bool LoadParam(string param_name, std::vector<std::vector<double>> &param_value, 
                      std::vector<std::vector<double>> default_value = {}, bool required = false) {
            XmlRpc::XmlRpcValue xml_value;
            if (nh_.getParam(param_name, xml_value)) {
                if (xml_value.getType() == XmlRpc::XmlRpcValue::TypeArray) {
                    param_value.clear();
                    for (int i = 0; i < xml_value.size(); ++i) {
                        XmlRpc::XmlRpcValue inner = xml_value[i];
                        if (inner.getType() == XmlRpc::XmlRpcValue::TypeArray) {
                            std::vector<double> tmp;
                            for (int j = 0; j < inner.size(); ++j) {
                                tmp.push_back(static_cast<double>(inner[j]));
                            }
                            param_value.push_back(tmp);
                        }
                    }
                }
                return true;
            }
            return false;
        }

        bool LoadWaypoints() {
            XmlRpc::XmlRpcValue waypoints_tmp;
            if (!nh_.getParam("astar/waypoints", waypoints_tmp)) {
                ROS_ERROR("Failed to get waypoints parameter");
                return false;
            }

            if (waypoints_tmp.getType() != XmlRpc::XmlRpcValue::TypeArray) {
                ROS_ERROR("Waypoints parameter is not an array");
                return false;
            }

            waypoints.clear();
            for (int i = 0; i < waypoints_tmp.size(); ++i) {
                XmlRpc::XmlRpcValue point = waypoints_tmp[i];
                if (point.getType() == XmlRpc::XmlRpcValue::TypeArray && point.size() == 3) {
                    waypoints.emplace_back(
                        static_cast<double>(point[0]),
                        static_cast<double>(point[1]),
                        static_cast<double>(point[2])
                    );
                    ROS_INFO("Added waypoint %d: [%f, %f, %f]", 
                            i,
                            static_cast<double>(point[0]),
                            static_cast<double>(point[1]),
                            static_cast<double>(point[2]));
                } else {
                    ROS_WARN("Invalid waypoint format at index %d", i);
                }
            }
            ROS_INFO("Total waypoints loaded: %zu", waypoints.size());
            return true;
        }

    public:
        bool visualize_process_en;
        bool allow_diag;
        Vec3i map_voxel_num, map_size_i;
        int heu_type;
        Vec3f example_start;
        Vec3f example_goal;
        std::vector<Vec3f> waypoints;
        double arrival_threshold;

        Config(const ros::NodeHandle & nh) : nh_(nh) {
            vector<int> vox;
            LoadParam("astar/visualize_process_en", visualize_process_en, false);
            LoadParam("astar/map_voxel_num", vox, vox);
            if(vox.size() == 3) {
                map_voxel_num = Vec3i(vox[0], vox[1], vox[2]);
                vox.clear();
            }
            LoadParam("astar/allow_diag", allow_diag, false);
            LoadParam("astar/heu_type", heu_type, 0);

            vector<double> tmp;
            LoadParam("astar/example_start", tmp, tmp);
            if(tmp.size() == 3) {
                example_start = Vec3f(tmp[0], tmp[1], tmp[2]);
                tmp.clear();
            }
            LoadParam("astar/example_goal", tmp, tmp);
            if(tmp.size() == 3) {
                example_goal = Vec3f(tmp[0], tmp[1], tmp[2]);
                tmp.clear();
            }

            map_size_i = map_voxel_num / 2;
            map_voxel_num = map_size_i * 2 + rog_map::Vec3i::Constant(1);

            // 加载标定点
            if (!LoadWaypoints()) {
                ROS_ERROR("Failed to load waypoints");
            }

            LoadParam("astar/arrival_threshold", arrival_threshold, 0.5);
        }
    };
}

#endif
