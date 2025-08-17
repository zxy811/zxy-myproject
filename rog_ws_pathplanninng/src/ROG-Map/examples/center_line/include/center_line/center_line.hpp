/**
 * @file center_line.hpp
 * @brief 葡萄藤行间中心线提取算法
 */

#pragma once

#include <Eigen/Core>
#include <memory>
#include <vector>

#include "rog_map/rog_map.h"
#include "rog_map/esdf_map.h"
using rog_map::ROGMap;
namespace center_line {

/**
 * @brief 葡萄藤行间中心线提取器类
 */
class CenterLine {
public:
    /**
     * @brief 构造函数
     * @param rog_map ROG-Map实例的共享指针
     * @param max_corridor_width 最大走廊宽度
     * @param min_confidence_threshold 最小置信度阈值
     */
    CenterLine(const std::shared_ptr<ROGMap>& rog_map,
              double max_corridor_width = 3.0,
              double min_confidence_threshold = 0.6);
    
    /**
     * @brief 析构函数
     */
    ~CenterLine() = default;
    
    /**
     * @brief 提取中心线
     * @param start_point 起点坐标
     * @param direction 初始方向
     * @param length 中心线长度
     * @param step_size 采样步长
     * @return 中心线点序列
     */
    std::vector<Eigen::Vector3d> extractCenterline(
        const Eigen::Vector3d& start_point,
        const Eigen::Vector3d& direction,
        double length,
        double step_size = 0.2);
    
    /**
     * @brief 设置最大走廊宽度
     * @param width 走廊宽度(米)
     */
    void setMaxCorridorWidth(double width) { max_corridor_width_ = width; }
    
    /**
     * @brief 设置最小置信度阈值
     * @param threshold 置信度阈值(0-1)
     */
    void setMinConfidenceThreshold(double threshold) { min_confidence_threshold_ = threshold; }
    
private:
    // 走廊分析结果结构体
    struct CorridorAnalysisResult {
        Eigen::Vector3d center_point;  // 走廊中心点
        double corridor_width;         // 走廊宽度
        double corridor_score;         // 走廊特征分数
        double confidence;             // 结果置信度
    };
    
    // ROG-Map实例
    std::shared_ptr<ROGMap> rog_map_;
    
    // 参数
    double max_corridor_width_;
    double min_confidence_threshold_;
    
    // 内部方法
    Eigen::Vector3d findBestCenterPoint(const Eigen::Vector3d& position, const Eigen::Vector3d& direction);
    CorridorAnalysisResult analyzeDirectionCorridor(const Eigen::Vector3d& position, const Eigen::Vector3d& direction);
    CorridorAnalysisResult analyzeESDFProfile(const std::vector<double>& esdf_values, const std::vector<Eigen::Vector3d>& sample_points);
    std::vector<double> smoothData(const std::vector<double>& data, int window_size);
    double calculateCorridorScore(const std::vector<double>& values, int center_idx);
    double calculateSymmetryScore(const std::vector<double>& values, int center_idx);
    double calculateBoundaryScore(const std::vector<double>& values, int center_idx);
    double estimateCorridorWidth(const std::vector<double>& values, int center_idx);
    double calculateConfidence(const std::vector<double>& values, int center_idx, double corridor_width);
    Eigen::Vector3d predictNextCenter(const std::vector<Eigen::Vector3d>& centerline, const Eigen::Vector3d& direction);
    std::vector<Eigen::Vector3d> smoothCenterline(const std::vector<Eigen::Vector3d>& centerline);
};

}  // namespace center_line
