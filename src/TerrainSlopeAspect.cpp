#include "TerrainSlopeAspect.hpp"
#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <cmath>

TerrainSlopeAspect::TerrainSlopeAspect(const cv::Mat& dem_m,const std::string& root_out,double grid_size,double theta_max,double inf_cost)
: dem_m_(dem_m), grid_size_(grid_size),theta_max_(theta_max), inf_cost_(inf_cost), root_out_(root_out)
{
	namespace fs = std::filesystem;
	const std::string root = root_out + "/TerrainSlopeAspect";
	fs::create_directories(root);
	out_txt_dir_ = root + "/out_txt_file";
	out_img_dir_ = root + "/out_image_file";
	fs::create_directories(out_txt_dir_);
	fs::create_directories(out_img_dir_);

	computeSlopeAspect_();
	buildCost_();
	export_file();
}

/* ----------  计算坡度 / 坡向 ---------- */
void TerrainSlopeAspect::computeSlopeAspect_()
{
	const int rows = dem_m_.rows, cols = dem_m_.cols;
	slope_deg_.create(rows, cols, CV_64FC1);
	aspect_deg_.create(rows, cols, CV_64FC1);
	slope_deg_ = cv::Scalar(90.0);
	aspect_deg_ = cv::Scalar(0.0);
	if (rows < 3 || cols < 3) {
		return;
	}

	const double denom = 6.0 * grid_size_;
	for (int y = 1; y < rows - 1; ++y) {
		const double* rN = dem_m_.ptr<double>(y - 1);
		const double* rC = dem_m_.ptr<double>(y);
		const double* rS = dem_m_.ptr<double>(y + 1);
		double* srow = slope_deg_.ptr<double>(y);
		double* arow = aspect_deg_.ptr<double>(y);
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
			double atan_term = (std::abs(fx) < 1e-15) ?
				((fy >= 0.0) ? (kPi / 2.0) : (-kPi / 2.0)) : std::atan(fy / fx);
			double A = 270.0 + rad2Deg_(atan_term) - 90.0 * sgn_fx;
			arow[x] = wrap360_(A);
		}
	}
}

/* ---------- 计算代价 / 障碍 ---------- */
void TerrainSlopeAspect::buildCost_()
{
	const int rows = slope_deg_.rows;
	const int cols = slope_deg_.cols;
	cost_distance_.create(rows, cols, CV_64FC1);
	cost_slope_.create(rows, cols, CV_64FC1);
	obstacle_.create(rows, cols, CV_8UC1);

	for (int y = 0; y < rows; ++y) {
		const double* s = slope_deg_.ptr<double>(y);
		double* d = cost_distance_.ptr<double>(y);
		double* p = cost_slope_.ptr<double>(y);
		unsigned char* o = obstacle_.ptr<unsigned char>(y);
		for (int x = 0; x < cols; ++x) {
			const double theta = s[x];
			if (theta >= theta_max_) {          // 统一障碍
				d[x] = inf_cost_;
				p[x] = inf_cost_;
				o[x] = 1;
				continue;
			}
			o[x] = 0;
			d[x] = 0.0;                     // 距离最短
			p[x] = theta / theta_max_;      // 坡度代价最小
		}
	}
}
/* ----------  导出 txt ---------- */
void TerrainSlopeAspect::export_file()
{
	std::cout << "\nOutputs:\n";
	// 5 个 txt
	exportText_(slope_deg_, out_txt_dir_ + "/slope_deg.txt", 3);
	std::cout << "  " << out_txt_dir_ << "/slope_deg.txt\n";
	exportText_(aspect_deg_, out_txt_dir_ + "/aspect_deg.txt", 3);
	std::cout << "  " << out_txt_dir_ << "/aspect_deg.txt\n";
	exportText_(obstacle_, out_txt_dir_ + "/obstacle.txt");
	std::cout << "  " << out_txt_dir_ << "/obstacle.txt\n";
	exportText_(cost_distance_, out_txt_dir_ + "/cost_distance.txt", 3);
	std::cout << "  " << out_txt_dir_ << "/cost_distance.txt\n";
	exportText_(cost_slope_, out_txt_dir_ + "/cost_slope.txt", 3);
	std::cout << "  " << out_txt_dir_ << "/cost_slope.txt\n";

	// 2 个 png
	savePng_(out_img_dir_ + "/slope_deg.png",visualizePNG_(slope_deg_, theta_max_));
	std::cout << "  " << out_img_dir_ << "/slope_deg.png\n";
	savePng_(out_img_dir_ + "/obstacle.png",visualizePNG_(obstacle_, theta_max_));
	std::cout << "  " << out_img_dir_ << "/obstacle.png\n";
}

/* =================================================================== */
/* =                          工具函数                                  = */
/* =================================================================== */

double TerrainSlopeAspect::rad2Deg_(double r) { return r * 180.0 / kPi; }
double TerrainSlopeAspect::wrap360_(double a)
{
	a = std::fmod(a, 360.0);
	if (a < 0.0) a += 360.0;
	return a;
}

void TerrainSlopeAspect::exportText_(const cv::Mat& m, const std::string& path, int precision)
{
	if (m.empty() || m.channels() != 1)
		throw std::runtime_error("exportText_: input must be single-channel CV_64FC1 or CV_8UC1.");
	namespace fs = std::filesystem;
	fs::path p(path);
	if (!p.parent_path().empty()) fs::create_directories(p.parent_path());
	std::ofstream ofs(path);
	if (!ofs.is_open()) throw std::runtime_error("exportText_: failed to open " + path);

	const int rows = m.rows;
	const int cols = m.cols;
	if (m.type() == CV_64FC1) {
		ofs << std::fixed << std::setprecision(precision);
		for (int r = 0; r < rows; ++r) {
			const double* row = m.ptr<double>(r);
			for (int c = 0; c < cols; ++c) ofs << row[c] << (c + 1 < cols ? ' ' : '\n');
		}
	}
	else if (m.type() == CV_8UC1) {
		for (int r = 0; r < rows; ++r) {
			const unsigned char* row = m.ptr<unsigned char>(r);
			for (int c = 0; c < cols; ++c) ofs << static_cast<int>(row[c]) << (c + 1 < cols ? ' ' : '\n');
		}
	}
	else {
		throw std::runtime_error("exportText_: unsupported matrix type.");
	}
}

cv::Mat TerrainSlopeAspect::visualizePNG_(const cv::Mat& m, double theta_max)
{
	if (m.empty() || m.channels() != 1)
		throw std::runtime_error("visualizePNG_: input must be single-channel CV_64FC1 or CV_8UC1.");
	const int rows = m.rows;
	const int cols = m.cols;
	cv::Mat out(rows, cols, CV_8UC1);

	if (m.type() == CV_64FC1) {                      // 坡度图
		const double tMax = theta_max <= 0.0 ? 20.0 : theta_max;
		for (int r = 0; r < rows; ++r) {
			const double* src = m.ptr<double>(r);
			unsigned char* dst = out.ptr<unsigned char>(r);
			for (int c = 0; c < cols; ++c) {
				double v = src[c];
				if (v < 0.0) v = 0.0;
				if (v >= tMax) dst[c] = 255;
				else {
					int pix = static_cast<int>((v / tMax) * 255.0 + 0.5);
					dst[c] = static_cast<unsigned char>(std::clamp(pix, 0, 255));
				}
			}
		}
	}
	else if (m.type() == CV_8UC1) {                // 障碍图
		for (int r = 0; r < rows; ++r) {
			const unsigned char* src = m.ptr<unsigned char>(r);
			unsigned char* dst = out.ptr<unsigned char>(r);
			for (int c = 0; c < cols; ++c) dst[c] = src[c] ? 255 : 0;
		}
	}
	else {
		throw std::runtime_error("visualizePNG_: unsupported matrix type.");
	}
	return out;
}

void TerrainSlopeAspect::savePng_(const std::string& path, const cv::Mat& img)
{
	namespace fs = std::filesystem;
	fs::path p(path);
	if (!p.parent_path().empty()) fs::create_directories(p.parent_path());
	if (!cv::imwrite(path, img)) throw std::runtime_error("savePng_ failed: " + path);
}