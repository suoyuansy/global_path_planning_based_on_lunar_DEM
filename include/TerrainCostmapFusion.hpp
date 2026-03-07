#pragma once
#include <opencv2/core.hpp>
#include <string>

#include "TerrainSlopeAspect.hpp"
#include "TerrainRoughness.hpp"
#include "TerrainStepEdge.hpp"
#include "TerrainObstacleExpand.hpp"

class TerrainCostmapFusion {
public:
	TerrainCostmapFusion(const TerrainSlopeAspect& tsa,const TerrainRoughness& rough,const TerrainStepEdge& step,const TerrainObstacleExpand& expand,const std::string& root_out,bool export_file_flag = true);
    // 新增：供外部直接获取最终代价地图
    const cv::Mat& costmap_merge_expand() const { return costmap_merge_expand_; }

private:
	std::string root_out_;
	std::string costmap_dir_;
	std::string merge_dir_;
    // 新增：缓存最终需要的代价地图
    cv::Mat costmap_merge_expand_;
    /* ---------- 内部算法 ---------- */
    cv::Mat createCostmap_(const cv::Mat& cost, const cv::Mat& mask) const;
    cv::Mat mergeCost_(const cv::Mat& c1,const cv::Mat& c2,const cv::Mat& c3) const;
    void export_costmap(const cv::Mat& merge_cost_distance,const cv::Mat& merge_cost,const TerrainSlopeAspect& tsa,const TerrainRoughness& rough,const TerrainStepEdge& step,const TerrainObstacleExpand& expand);
    /* ---------- 工具函数 ---------- */
    static void exportText_(const cv::Mat& m, const std::string& path, int precision = 3);
};