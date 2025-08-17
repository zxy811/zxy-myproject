/**
 * @file center_line.cpp
 * @brief 葡萄藤行间中心线提取算法实现
 */

#include "center_line/center_line.hpp"
#include <algorithm>
#include <limits>
#include <cmath>

namespace center_line {

CenterLine::CenterLine(
    const std::shared_ptr<ROGMap>& rog_map,
    double max_corridor_width,
    double min_confidence_threshold)
    : rog_map_(rog_map),
      max_corridor_width_(max_corridor_width),
      min_confidence_threshold_(min_confidence_threshold) {
}

std::vector<Eigen::Vector3d> CenterLine::extractCenterline(
    const Eigen::Vector3d& start_point,
    const Eigen::Vector3d& direction,
    double length,
    double step_size) {
    
    std::vector<Eigen::Vector3d> centerline;
    
    // 首先找到起点附近的最佳中心点
    Eigen::Vector3d current_point = findBestCenterPoint(start_point, direction);
    centerline.push_back(current_point);
    
    Eigen::Vector3d current_dir = direction.normalized();
    double traveled_distance = 0.0;
    
    // 逐步生成中心线
    while (traveled_distance < length) {
        // 分析当前位置的横截面
        CorridorAnalysisResult analysis = analyzeDirectionCorridor(current_point, current_dir);
        
        // 更新中心点
        Eigen::Vector3d center_point;
        if (analysis.confidence > min_confidence_threshold_) {
            center_point = analysis.center_point;
        } else {
            // 如果可信度低，使用前向预测
            center_point = predictNextCenter(centerline, current_dir);
        }
        
        // 沿着方向前进一步
        Eigen::Vector3d next_point = center_point + step_size * current_dir;
        centerline.push_back(next_point);
        
        // 更新方向（使用平滑）
        if (centerline.size() >= 3) {
            Eigen::Vector3d last_dir = (centerline.back() - centerline[centerline.size()-2]).normalized();
            current_dir = (current_dir * 0.7 + last_dir * 0.3).normalized();
        }
        
        // 更新位置和距离
        current_point = next_point;
        traveled_distance += step_size;
    }
    
    // 平滑中心线
    return smoothCenterline(centerline);
}

Eigen::Vector3d CenterLine::findBestCenterPoint(const Eigen::Vector3d& position, const Eigen::Vector3d& direction) {
    // 获取横向方向（垂直于前进方向）
    Eigen::Vector3d up(0, 0, 1);
    Eigen::Vector3d lateral_dir = direction.cross(up).normalized();
    
    // 在横向上采样点
    std::vector<Eigen::Vector3d> sample_points;
    std::vector<double> esdf_values;
    
    // 采样参数
    double sampling_width = max_corridor_width_;
    int num_samples = 20;
    double step = sampling_width / num_samples;
    
    // 从左到右采样
    for (int i = 0; i <= num_samples; ++i) {
        double offset = -sampling_width/2 + i * step;
        Eigen::Vector3d sample_point = position + offset * lateral_dir;
        
        // 获取ESDF值
        double distance = 0.0;
        if (rog_map_->evaluateCoarseEDT(sample_point, distance)) {
            sample_points.push_back(sample_point);
            esdf_values.push_back(distance);
        }
    }
    
    // 如果没有有效点，直接返回当前位置
    if (sample_points.empty()) {
        return position;
    }
    
    // 分析ESDF值曲线，找到中心点
    CorridorAnalysisResult analysis = analyzeESDFProfile(esdf_values, sample_points);
    
    // 如果分析结果可靠，返回找到的中心点
    if (analysis.confidence > min_confidence_threshold_) {
        return analysis.center_point;
    }
    
    // 否则返回当前位置
    return position;
}

CenterLine::CorridorAnalysisResult CenterLine::analyzeDirectionCorridor(
    const Eigen::Vector3d& position,
    const Eigen::Vector3d& direction) {
    
    // 获取横向方向
    Eigen::Vector3d up(0, 0, 1);
    Eigen::Vector3d lateral_dir = direction.cross(up).normalized();
    
    // 在横向上采样
    std::vector<Eigen::Vector3d> sample_points;
    std::vector<double> esdf_values;
    
    // 采样参数
    double sampling_width = max_corridor_width_;
    int num_samples = 20;
    double step = sampling_width / num_samples;
    
    // 从左到右采样
    for (int i = 0; i <= num_samples; ++i) {
        double offset = -sampling_width/2 + i * step;
        Eigen::Vector3d sample_point = position + offset * lateral_dir;
        
        // 获取ESDF值
        double distance = 0.0;
        if (rog_map_->evaluateCoarseEDT(sample_point, distance)) {
            sample_points.push_back(sample_point);
            esdf_values.push_back(distance);
        }
    }
    
    // 如果样本点不足，返回低置信度结果
    if (sample_points.size() < 5) {
        CorridorAnalysisResult result;
        result.center_point = position;
        result.corridor_width = 0.0;
        result.corridor_score = 0.0;
        result.confidence = 0.0;
        return result;
    }
    
    // 分析ESDF值曲线
    return analyzeESDFProfile(esdf_values, sample_points);
}

CenterLine::CorridorAnalysisResult CenterLine::analyzeESDFProfile(
    const std::vector<double>& esdf_values,
    const std::vector<Eigen::Vector3d>& sample_points) {
    
    CorridorAnalysisResult result;
    
    // 如果样本点太少，返回低置信度结果
    if (esdf_values.size() < 5) {
        result.center_point = sample_points[0];  // 默认使用第一个点
        result.corridor_width = 0.0;
        result.corridor_score = 0.0;
        result.confidence = 0.0;
        return result;
    }
    
    // 平滑处理ESDF值（简单的移动平均）
    std::vector<double> smoothed_values = smoothData(esdf_values, 3);
    
    // 找出所有局部最大值点
    std::vector<int> local_maxima;
    for (size_t i = 1; i < smoothed_values.size() - 1; ++i) {
        if (smoothed_values[i] > smoothed_values[i-1] && 
            smoothed_values[i] > smoothed_values[i+1]) {
            local_maxima.push_back(i);
        }
    }
    
    // 如果没有局部最大值，使用全局最大值
    if (local_maxima.empty()) {
        auto max_it = std::max_element(smoothed_values.begin(), smoothed_values.end());
        int max_idx = std::distance(smoothed_values.begin(), max_it);
        local_maxima.push_back(max_idx);
    }
    
    // 分析每个局部最大值，找出最可能的走廊中心
    int best_idx = -1;
    double best_score = -1.0;
    
    for (int max_idx : local_maxima) {
        // 计算走廊特征得分
        double score = calculateCorridorScore(smoothed_values, max_idx);
        
        if (score > best_score) {
            best_score = score;
            best_idx = max_idx;
        }
    }
    
    // 计算走廊宽度
    double corridor_width = estimateCorridorWidth(smoothed_values, best_idx);
    
    // 填充结果
    result.center_point = sample_points[best_idx];
    result.corridor_width = corridor_width;
    result.corridor_score = best_score;
    result.confidence = calculateConfidence(smoothed_values, best_idx, corridor_width);
    
    return result;
}

std::vector<double> CenterLine::smoothData(const std::vector<double>& data, int window_size) {
    std::vector<double> smoothed = data;
    
    for (size_t i = 0; i < data.size(); ++i) {
        int count = 0;
        double sum = 0.0;
        
        for (int j = -window_size/2; j <= window_size/2; ++j) {
            int idx = i + j;
            if (idx >= 0 && idx < (int)data.size()) {
                sum += data[idx];
                count++;
            }
        }
        
        if (count > 0) {
            smoothed[i] = sum / count;
        }
    }
    
    return smoothed;
}

double CenterLine::calculateCorridorScore(const std::vector<double>& values, int center_idx) {
    // 1. 中心点的ESDF值
    double center_value = values[center_idx];
    
    // 2. 对称性得分
    double symmetry_score = calculateSymmetryScore(values, center_idx);
    
    // 3. 走廊边界得分
    double boundary_score = calculateBoundaryScore(values, center_idx);
    
    // 综合得分（权重可调整）
    return center_value * 0.4 + symmetry_score * 0.4 + boundary_score * 0.2;
}

double CenterLine::calculateSymmetryScore(const std::vector<double>& values, int center_idx) {
    int max_offset = std::min(center_idx, (int)values.size() - center_idx - 1);
    if (max_offset <= 0) return 0.0;
    
    double sum_diff = 0.0;
    for (int offset = 1; offset <= max_offset; ++offset) {
        double left_val = values[center_idx - offset];
        double right_val = values[center_idx + offset];
        sum_diff += std::abs(left_val - right_val);
    }
    
    // 归一化差异
    double avg_diff = sum_diff / max_offset;
    double max_val = values[center_idx];
    
    // 返回对称性得分（0-1之间，越大表示越对称）
    return 1.0 - std::min(1.0, avg_diff / (max_val + 1e-6));
}

double CenterLine::calculateBoundaryScore(const std::vector<double>& values, int center_idx) {
    // 检查左右两侧是否存在明显的边界（ESDF值下降）
    bool left_boundary = false;
    bool right_boundary = false;
    
    // 检查左侧
    for (int i = center_idx - 1; i >= 0; --i) {
        if (values[i] < values[center_idx] * 0.5) {  // 距离值降低到一半以下认为是边界
            left_boundary = true;
            break;
        }
    }
    
    // 检查右侧
    for (size_t i = center_idx + 1; i < values.size(); ++i) {
        if (values[i] < values[center_idx] * 0.5) {
            right_boundary = true;
            break;
        }
    }
    
    // 如果两侧都有边界，返回1.0，否则返回相应的得分
    if (left_boundary && right_boundary) return 1.0;
    if (left_boundary || right_boundary) return 0.5;
    return 0.2;  // 如果没有检测到明显边界，返回低分
}

double CenterLine::estimateCorridorWidth(const std::vector<double>& values, int center_idx) {
    double center_value = values[center_idx];
    double threshold = center_value * 0.5;  // 使用50%的中心值作为走廊边界
    
    // 向左搜索边界
    int left_idx = center_idx;
    for (int i = center_idx - 1; i >= 0; --i) {
        if (values[i] < threshold) {
            left_idx = i;
            break;
        }
    }
    
    // 向右搜索边界
    int right_idx = center_idx;
    for (size_t i = center_idx + 1; i < values.size(); ++i) {
        if (values[i] < threshold) {
            right_idx = i;
            break;
        }
    }
    
    // 计算宽度（这里需要知道采样点的实际距离）
    double width = (right_idx - left_idx) * (max_corridor_width_ / values.size());
    return width;
}

double CenterLine::calculateConfidence(const std::vector<double>& values, int center_idx, double corridor_width) {
    // 1. 中心值得分（距离障碍物越远越好）
    double center_value = values[center_idx];
    double distance_score = std::min(1.0, center_value / 2.0);  // 假设2米是理想距离
    
    // 2. 对称性得分
    double symmetry_score = calculateSymmetryScore(values, center_idx);
    
    // 3. 宽度得分（走廊宽度在合理范围内）
    double width_score;
    if (corridor_width < 0.5) {
        width_score = corridor_width / 0.5;  // 太窄
    } else if (corridor_width > 2.5) {
        width_score = 1.0 - std::min(1.0, (corridor_width - 2.5) / 2.5);  // 太宽
    } else {
        width_score = 1.0;  // 理想宽度
    }
    
    // 综合计算置信度
    return (distance_score * 0.4 + symmetry_score * 0.4 + width_score * 0.2);
}

Eigen::Vector3d CenterLine::predictNextCenter(
    const std::vector<Eigen::Vector3d>& centerline,
    const Eigen::Vector3d& direction) {
    
    if (centerline.empty()) {
        return Eigen::Vector3d::Zero();
    }
    
    if (centerline.size() == 1) {
        return centerline[0] + 0.2 * direction;
    }
    
    // 使用最近几个点的移动趋势进行预测
    int n = std::min(3, (int)centerline.size());
    Eigen::Vector3d trend(0, 0, 0);
    
    for (int i = 1; i < n; ++i) {
        trend += centerline[centerline.size()-i] - centerline[centerline.size()-i-1];
    }
    
    trend /= (n-1);
    return centerline.back() + trend;
}

std::vector<Eigen::Vector3d> CenterLine::smoothCenterline(const std::vector<Eigen::Vector3d>& centerline) {
    if (centerline.size() < 3) return centerline;
    
    std::vector<Eigen::Vector3d> smoothed;
    smoothed.push_back(centerline[0]);  // 保留起点
    
    for (size_t i = 1; i < centerline.size() - 1; ++i) {
        Eigen::Vector3d avg = (centerline[i-1] + centerline[i] + centerline[i+1]) / 3.0;
        smoothed.push_back(avg);
    }
    
    smoothed.push_back(centerline.back());  // 保留终点
    return smoothed;
}

}  // namespace center_line
