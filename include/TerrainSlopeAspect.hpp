#pragma once
#include <opencv2/core.hpp>
#include <string>
enum class PlanningStrategy { DistanceShortest, SlopeCostMin };
struct SlopeAspectResult {
	cv::Mat slope_deg;
	cv::Mat aspect_deg;
};
struct CostResult {
	cv::Mat cost;
	cv::Mat obstacle;   // 0 可通行 / 1 障碍
};
class TerrainSlopeAspect {
public:
	TerrainSlopeAspect(const cv::Mat& dem_m,const std::string& out_img_dir,const std::string& out_txt_dir,double grid_size = 1.0,double theta_max_deg = 20.0,double inf_cost = 1e10);
	/* ---- 只读成果 ---- */
	const cv::Mat& slope_deg()      const { return slope_deg_; }
	const cv::Mat& aspect_deg()     const { return aspect_deg_; }
	const cv::Mat& cost_distance()  const { return cost_distance_; }
	const cv::Mat& cost_slope()     const { return cost_slope_; }
	const cv::Mat& obstacle_dist()  const { return obstacle_dist_; }
	const cv::Mat& obstacle_slope() const { return obstacle_slope_; }
private:
	cv::Mat dem_m_;          // 输入高程（米）
	cv::Mat slope_deg_;      // 坡度（°）
	cv::Mat aspect_deg_;     // 坡向（°）
	cv::Mat cost_distance_;  // 距离最短策略代价
	cv::Mat cost_slope_;     // 坡度最小策略代价
	cv::Mat obstacle_dist_;  // 对应障碍
	cv::Mat obstacle_slope_; // 对应障碍
	double grid_size_;
	double theta_max_deg_;
	double inf_cost_;
	std::string out_img_dir_;
	std::string out_txt_dir_;

	/* ---------- 内部流程 ---------- */
	void computeSlopeAspect_();      // 坡度/坡向
	void buildCost_();               // 两种代价
	void exportTxt_() const;         // 导出 4 个 txt
	void exportPng_() const;         // 导出 4 个 png

	/* 原 TerrainDerivatives */
	static SlopeAspectResult computeSlopeAspect_3rdOrder_(const cv::Mat& dem, double g);

	/* 原 TerrainCost 实现 */
	static CostResult buildCostFromSlope_(const cv::Mat& slope_deg,PlanningStrategy strategy,double theta_max,double inf_cost);

	/* 原 TerrainIO 实现 */
	static void exportMat64ToText_(const cv::Mat& m64, const std::string& path, int precision);
	static cv::Mat visualizeSlope8U_(const cv::Mat& slope_deg, double theta_max);
	static cv::Mat visualizeCost8U_(const cv::Mat& cost64, const cv::Mat& obstacle8u, double inf_cost);
	static void saveGrayPng_(const std::string& path, const cv::Mat& img8u);

	/* 小工具 */
	static constexpr double kPi = 3.14159265358979323846;
	static double rad2Deg_(double r) { return r * 180.0 / kPi; }
	static double wrap360_(double a);
};