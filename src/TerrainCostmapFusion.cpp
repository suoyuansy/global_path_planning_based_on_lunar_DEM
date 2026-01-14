#include "TerrainCostmapFusion.hpp"
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>

TerrainCostmapFusion::TerrainCostmapFusion(const TerrainSlopeAspect& tsa,const TerrainRoughness& rough,const TerrainStepEdge& step,const TerrainObstacleExpand& expand,const std::string& root_out)
: root_out_(root_out) 
{
	namespace fs = std::filesystem;
	costmap_dir_ = root_out_ + "/costmap";
	merge_dir_ = root_out_ + "/Merge";

	fs::create_directories(costmap_dir_);
	fs::create_directories(merge_dir_);
	/* 计算两种融合代价 */
	cv::Mat merge_cost_distance = mergeCost_(rough.cost_distance(),tsa.cost_distance(),step.cost_distance());
	cv::Mat merge_cost = mergeCost_(rough.cost_rollover(),tsa.cost_slope(),step.cost_step());

	/* 生成 16 种 costmap 并导出 */
	export_costmap(merge_cost_distance, merge_cost, tsa, rough, step, expand);
}

/* ---------- 生成单张 costmap ---------- */
cv::Mat TerrainCostmapFusion::createCostmap_(const cv::Mat& cost,const cv::Mat& mask) const 
{
	if (cost.size() != mask.size() ||cost.type() != CV_64FC1 ||mask.type() != CV_8UC1)
		throw std::runtime_error(
			"createCostmap_: size mismatch or wrong type (cost:CV_64FC1, mask:CV_8UC1).");
	cv::Mat ans = cost.clone();
	ans.setTo(1.0, mask);
	return ans;
}

/* ---------- 三矩阵等权融合 ---------- */
cv::Mat TerrainCostmapFusion::mergeCost_(const cv::Mat& c1,const cv::Mat& c2,const cv::Mat& c3) const 
{
	if (c1.size() != c2.size() || c2.size() != c3.size() ||c1.type() != CV_64FC1 || c2.type() != CV_64FC1 || c3.type() != CV_64FC1)
		throw std::runtime_error(
			"mergeCost_: matrices must be same size and CV_64FC1.");
	return (c1 + c2 + c3) / 3.0;
}

/* ---------- 导出 16 种 costmap ---------- */
void TerrainCostmapFusion::export_costmap(const cv::Mat& merge_cost_distance, const cv::Mat& merge_cost, const TerrainSlopeAspect& tsa, const TerrainRoughness& rough, const TerrainStepEdge& step, const TerrainObstacleExpand& expand)
{
	std::cout << "\nOutputs:\n";
	/* 先将融合中间结果导出到 Merge 目录 */
	exportText_(merge_cost_distance, merge_dir_ + "/merge_cost_distance.txt", 3);
	std::cout << "  " << merge_dir_ << "/merge_cost_distance.txt\n";
	exportText_(merge_cost, merge_dir_ + "/merge_cost.txt", 3);
	std::cout << "  " << merge_dir_ << "/merge_cost.txt\n";
	/* 再将16种策略导出 */
	exportText_(createCostmap_(tsa.cost_distance(), tsa.obstacle()), costmap_dir_ + "/costmap_TerrainSlope_distance.txt", 3);
	std::cout << "  " << costmap_dir_ << "/costmap_TerrainSlope_distance.txt\n";
	exportText_(createCostmap_(tsa.cost_slope(), tsa.obstacle()),costmap_dir_ + "/costmap_TerrainSlope.txt", 3);
	std::cout << "  " << costmap_dir_ << "/costmap_TerrainSlope.txt\n";
	exportText_(createCostmap_(tsa.cost_slope(), expand.slope_expand()),costmap_dir_ + "/costmap_TerrainSlope_expand.txt", 3);
	std::cout << "  " << costmap_dir_ << "/costmap_TerrainSlope_expand.txt\n";
	exportText_(createCostmap_(tsa.cost_distance(), expand.slope_expand()),costmap_dir_ + "/costmap_TerrainSlope_expand_distance.txt", 3);
	std::cout << "  " << costmap_dir_ << "/costmap_TerrainSlope_expand_distance.txt\n";
	exportText_(createCostmap_(rough.cost_distance(), rough.obstacle()),costmap_dir_ + "/costmap_TerrainRoughness_distance.txt", 3);
	std::cout << "  " << costmap_dir_ << "/costmap_TerrainRoughness_distance.txt\n";
	exportText_(createCostmap_(rough.cost_rollover(), rough.obstacle()),costmap_dir_ + "/costmap_TerrainRoughness.txt", 3);
	std::cout << "  " << costmap_dir_ << "/costmap_TerrainRoughness.txt\n";
	exportText_(createCostmap_(rough.cost_rollover(), expand.rough_expand()),costmap_dir_ + "/costmap_TerrainRoughness_expand.txt", 3);
	std::cout << "  " << costmap_dir_ << "/costmap_TerrainRoughness_expand.txt\n";
	exportText_(createCostmap_(rough.cost_distance(), expand.rough_expand()),costmap_dir_ + "/costmap_TerrainRoughness_expand_distance.txt", 3);
	std::cout << "  " << costmap_dir_ << "/costmap_TerrainRoughness_expand_distance.txt\n";
	exportText_(createCostmap_(step.cost_distance(), step.step_obstacle()),costmap_dir_ + "/costmap_TerrainStepEdge_distance.txt", 3);
	std::cout << "  " << costmap_dir_ << "/costmap_TerrainStepEdge_distance.txt\n";
	exportText_(createCostmap_(step.cost_step(), step.step_obstacle()),costmap_dir_ + "/costmap_TerrainStepEdge.txt", 3);
	std::cout << "  " << costmap_dir_ << "/costmap_TerrainStepEdge.txt\n";
	exportText_(createCostmap_(step.cost_step(), expand.step_expand()),costmap_dir_ + "/costmap_TerrainStepEdge_expand.txt", 3);
	std::cout << "  " << costmap_dir_ << "/costmap_TerrainStepEdge_expand.txt\n";
	exportText_(createCostmap_(step.cost_distance(), expand.step_expand()),costmap_dir_ + "/costmap_TerrainStepEdge_expand_distance.txt", 3);
	std::cout << "  " << costmap_dir_ << "/costmap_TerrainStepEdge_expand_distance.txt\n";
	exportText_(createCostmap_(merge_cost_distance, expand.union_mask()),costmap_dir_ + "/costmap_Merge_distance.txt", 3);
	std::cout << "  " << costmap_dir_ << "/costmap_Merge_distance.txt\n";
	exportText_(createCostmap_(merge_cost, expand.union_mask()),costmap_dir_ + "/costmap_Merge.txt", 3);
	std::cout << "  " << costmap_dir_ << "/costmap_Merge.txt\n";
	exportText_(createCostmap_(merge_cost, expand.union_expand()),costmap_dir_ + "/costmap_Merge_expand.txt", 3);
	std::cout << "  " << costmap_dir_ << "/costmap_Merge_expand.txt\n";
	exportText_(createCostmap_(merge_cost_distance, expand.union_expand()),costmap_dir_ + "/costmap_Merge_expand_distance.txt", 3);
	std::cout << "  " << costmap_dir_ << "/costmap_Merge_expand_distance.txt\n";
}

/* ---------- 统一导出格式 ---------- */
void TerrainCostmapFusion::exportText_(const cv::Mat& m,const std::string& path,int precision) 
{
	if (m.empty() || m.channels() != 1 || m.type() != CV_64FC1)
		throw std::runtime_error("exportText_: input must be single-channel CV_64FC1.");
	std::filesystem::create_directories(std::filesystem::path(path).parent_path());
	std::ofstream ofs(path);
	if (!ofs.is_open()) throw std::runtime_error("exportText_: failed to open " + path);
	ofs << std::fixed << std::setprecision(precision);
	for (int r = 0; r < m.rows; ++r) {
		const double* row = m.ptr<double>(r);  
		for (int c = 0; c < m.cols; ++c) ofs << row[c] << (c + 1 < m.cols ? ' ' : '\n');
	}
}
