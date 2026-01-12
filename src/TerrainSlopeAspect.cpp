#include "TerrainSlopeAspect.hpp"
#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <algorithm>
#include <cmath>

TerrainSlopeAspect::TerrainSlopeAspect(const cv::Mat& dem_m,const std::string& out_img_dir,const std::string& out_txt_dir,double grid_size,double theta_max_deg,double inf_cost)
	: dem_m_(dem_m),// 把参数直接初始化给成员变量
	grid_size_(grid_size),
	theta_max_deg_(theta_max_deg),
	inf_cost_(inf_cost),
	out_img_dir_(out_img_dir),
	out_txt_dir_(out_txt_dir)
{
	namespace fs = std::filesystem;
	fs::create_directories(out_img_dir_);
	fs::create_directories(out_txt_dir_);

	computeSlopeAspect_();
	buildCost_();
	exportTxt_();
	exportPng_();
}

/* ---------- 1. 坡度 / 坡向 ---------- */
void TerrainSlopeAspect::computeSlopeAspect_()
{
	auto sa = computeSlopeAspect_3rdOrder_(dem_m_, grid_size_);
	slope_deg_ = sa.slope_deg;
	aspect_deg_ = sa.aspect_deg;
}

/* ---------- 2. 代价 / 障碍 ---------- */
void TerrainSlopeAspect::buildCost_()
{
	auto dist = buildCostFromSlope_(slope_deg_, PlanningStrategy::DistanceShortest, theta_max_deg_, inf_cost_);
	auto slope = buildCostFromSlope_(slope_deg_, PlanningStrategy::SlopeCostMin, theta_max_deg_, inf_cost_);
	cost_distance_ = dist.cost;
	obstacle_dist_ = dist.obstacle;
	cost_slope_ = slope.cost;
	obstacle_slope_ = slope.obstacle;
}
/* ---------- 3. 导出 txt ---------- */
void TerrainSlopeAspect::exportTxt_() const
{
	exportMat64ToText_(slope_deg_, out_txt_dir_ + "/slope_deg.txt", 3);
	exportMat64ToText_(aspect_deg_, out_txt_dir_ + "/aspect_deg.txt", 3);
	exportMat64ToText_(cost_distance_, out_txt_dir_ + "/cost_distance.txt", 3);
	exportMat64ToText_(cost_slope_, out_txt_dir_ + "/cost_slope.txt", 3);
}
/* ---------- 4. 导出 png ---------- */
void TerrainSlopeAspect::exportPng_() const
{
	cv::Mat slope_img = visualizeSlope8U_(slope_deg_, theta_max_deg_);
	cv::Mat cost_dist_img = visualizeCost8U_(cost_distance_, obstacle_dist_, inf_cost_);
	cv::Mat cost_slope_img = visualizeCost8U_(cost_slope_, obstacle_slope_, inf_cost_);
	saveGrayPng_(out_img_dir_ + "/slope_deg.png", slope_img);
	saveGrayPng_(out_img_dir_ + "/cost_distance.png", cost_dist_img);
	saveGrayPng_(out_img_dir_ + "/cost_slope.png", cost_slope_img);

	std::cout << "\nOutputs:\n";
	std::cout << "  " << out_txt_dir_ << "/slope_deg.txt\n";
	std::cout << "  " << out_txt_dir_ << "/aspect_deg.txt\n";
	std::cout << "  " << out_txt_dir_ << "/cost_distance.txt\n";
	std::cout << "  " << out_txt_dir_ << "/cost_slope.txt\n";
	std::cout << "  " << out_img_dir_ << "/slope_deg.png\n";
	std::cout << "  " << out_img_dir_ << "/cost_distance.png\n";
	std::cout << "  " << out_img_dir_ << "/cost_slope.png\n";
}

/* =================================================================== /
/ =                     原 TerrainDerivatives 实现                     = /
/ =================================================================== */
SlopeAspectResult TerrainSlopeAspect::computeSlopeAspect_3rdOrder_(const cv::Mat& dem, double g)
{
	if (dem.empty() || dem.type() != CV_64FC1 || dem.channels() != 1)
		throw std::runtime_error("computeSlopeAspect_3rdOrder_: dem must be CV_64FC1 single-channel.");
	if (g <= 0.0) throw std::runtime_error("computeSlopeAspect_3rdOrder_: g must be > 0.");
	const int rows = dem.rows;
	const int cols = dem.cols;

	cv::Mat slope(rows, cols, CV_64FC1, cv::Scalar(90.0));
	cv::Mat aspect(rows, cols, CV_64FC1, cv::Scalar(0.0));

	if (rows < 3 || cols < 3) return { slope, aspect };

	const double denom = 6.0 * g;
	for (int y = 1; y < rows - 1; ++y) {
		const double* rN = dem.ptr<double>(y - 1);
		const double* rC = dem.ptr<double>(y);
		const double* rS = dem.ptr<double>(y + 1);
		double* srow = slope.ptr<double>(y);
		double* arow = aspect.ptr<double>(y);
		for (int x = 1; x < cols - 1; ++x) {
			const double Z1 = rS[x - 1], Z2 = rS[x], Z3 = rS[x + 1];
			const double Z4 = rC[x - 1], Z6 = rC[x + 1];
			const double Z7 = rN[x - 1], Z8 = rN[x], Z9 = rN[x + 1];

			const double fx = ((Z7 - Z1) + (Z8 - Z2) + (Z9 - Z3)) / denom;
			const double fy = ((Z3 - Z1) + (Z6 - Z4) + (Z9 - Z7)) / denom;

			const double grad = std::sqrt(fx * fx + fy * fy);
			srow[x] = rad2Deg_(std::atan(grad));

			if (grad < 1e-12) { arow[x] = 0.0; continue; }

			const double sgn_fx = (fx > 0.0) ? 1.0 : (fx < 0.0 ? -1.0 : 1.0);
			double atan_term;
			if (std::abs(fx) < 1e-15) atan_term = (fy >= 0.0) ? (kPi / 2.0) : (-kPi / 2.0);
			else atan_term = std::atan(fy / fx);

			double A = 270.0 + rad2Deg_(atan_term) - 90.0 * sgn_fx;
			arow[x] = wrap360_(A);
		}
	}
	return { slope, aspect };
}

/* =================================================================== /
/ =                         原 TerrainCost 实现                        = /
/ =================================================================== */
CostResult TerrainSlopeAspect::buildCostFromSlope_(const cv::Mat& slope_deg,PlanningStrategy strategy,double theta_max,double inf_cost)
{
	if (slope_deg.empty() || slope_deg.type() != CV_64FC1 || slope_deg.channels() != 1)
		throw std::runtime_error("buildCostFromSlope_: slope_deg must be CV_64FC1 single-channel.");

	if (theta_max <= 0.0) throw std::runtime_error("buildCostFromSlope_: theta_max must be > 0.");
	cv::Mat cost(slope_deg.size(), CV_64FC1, cv::Scalar(0.0));
	cv::Mat obs(slope_deg.size(), CV_8UC1, cv::Scalar(0));

	for (int y = 0; y < slope_deg.rows; ++y) {
		const double* s = slope_deg.ptr<double>(y);
		double* c = cost.ptr<double>(y);
		unsigned char* o = obs.ptr<unsigned char>(y);
		for (int x = 0; x < slope_deg.cols; ++x) {
			const double theta = s[x];
			if (theta >= theta_max) {
				c[x] = inf_cost;
				o[x] = 1;
				continue;
			}
			if (strategy == PlanningStrategy::DistanceShortest) c[x] = 0.0;
			else c[x] = theta / theta_max; // 0~1
		}
	}
	return { cost, obs };
}

/* =================================================================== */
/* =                         原 TerrainIO 实现                          = */
/* =================================================================== */
void TerrainSlopeAspect::exportMat64ToText_(const cv::Mat& m64, const std::string& path, int precision)
{
	if (m64.empty() || m64.channels() != 1)
		throw std::runtime_error("exportMat64ToText_: input must be CV_64FC1 single-channel.");
	namespace fs = std::filesystem;
	fs::path p(path);
	if (!p.parent_path().empty()) fs::create_directories(p.parent_path());
	std::ofstream ofs(path);
	if (!ofs.is_open()) throw std::runtime_error("exportMat64ToText_: failed to open " + path);
	ofs << std::fixed << std::setprecision(precision);
	for (int r = 0; r < m64.rows; ++r) {
		const double* row = m64.ptr<double>(r);          // ← 修复指针类型
		for (int c = 0; c < m64.cols; ++c) {
			ofs << row[c] << (c + 1 < m64.cols ? ' ' : '\n');
		}
	}
}

cv::Mat TerrainSlopeAspect::visualizeSlope8U_(const cv::Mat& slope_deg, double theta_max)
{
	if (slope_deg.empty() || slope_deg.type() != CV_64FC1 || slope_deg.channels() != 1)
		throw std::runtime_error("visualizeSlope8U_: slope_deg must be CV_64FC1 single-channel.");
	if (theta_max <= 0.0) theta_max = 20.0;
	cv::Mat out(slope_deg.size(), CV_8UC1, cv::Scalar(0));
	for (int y = 0; y < slope_deg.rows; ++y) {
		const double* s = slope_deg.ptr<double>(y);
		unsigned char* o = out.ptr<unsigned char>(y);
		for (int x = 0; x < slope_deg.cols; ++x) {
			double v = s[x];
			if (v < 0.0) v = 0.0;
			if (v >= theta_max) o[x] = 255;
			else {
				int pix = static_cast<int>((v / theta_max) * 255.0 + 0.5);
				o[x] = static_cast<unsigned char>(std::clamp(pix, 0, 255));
			}
		}
	}
	return out;
}

cv::Mat TerrainSlopeAspect::visualizeCost8U_(const cv::Mat& cost64, const cv::Mat& obstacle8u, double inf_cost)
{
	if (cost64.empty() || cost64.type() != CV_64FC1 || cost64.channels() != 1)
		throw std::runtime_error("visualizeCost8U_: cost64 must be CV_64FC1 single-channel.");
	if (obstacle8u.empty() || obstacle8u.type() != CV_8UC1 || obstacle8u.channels() != 1)
		throw std::runtime_error("visualizeCost8U_: obstacle8u must be CV_8UC1 single-channel.");
	if (cost64.size() != obstacle8u.size())
		throw std::runtime_error("visualizeCost8U_: size mismatch.");
	cv::Mat out(cost64.size(), CV_8UC1, cv::Scalar(0));
	for (int y = 0; y < cost64.rows; ++y) {
		const double* c = cost64.ptr<double>(y);
		const unsigned char* o = obstacle8u.ptr<unsigned char>(y);
		unsigned char* dst = out.ptr<unsigned char>(y);
		for (int x = 0; x < cost64.cols; ++x) {
			if (o[x] != 0 || c[x] >= inf_cost * 0.5) dst[x] = 255;
			else {
				double v = c[x];
				if (v < 0.0) v = 0.0; 
				if (v > 1.0) v = 1.0;
				int pix = static_cast<int>(v * 255.0 + 0.5);
				dst[x] = static_cast<unsigned char>(std::clamp(pix, 0, 255));
			}
		}
	}
	return out;
}

void TerrainSlopeAspect::saveGrayPng_(const std::string& path, const cv::Mat& img8u)
{
	if (img8u.empty() || img8u.type() != CV_8UC1)
		throw std::runtime_error("saveGrayPng_: img must be CV_8UC1.");
	namespace fs = std::filesystem;
	fs::path p(path);
	if (!p.parent_path().empty()) fs::create_directories(p.parent_path());
	if (!cv::imwrite(path, img8u)) throw std::runtime_error("saveGrayPng_ failed: " + path);
}

double TerrainSlopeAspect::wrap360_(double a)
{
	a = std::fmod(a, 360.0);
	if (a < 0.0) a += 360.0;
	return a;
}