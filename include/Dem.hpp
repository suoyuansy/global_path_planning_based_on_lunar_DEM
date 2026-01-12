#pragma once
#include <opencv2/core.hpp>
#include <string>


class Dem {
public:
	Dem(const std::string& tiff_path,const std::string& out_img_dir,const std::string& out_txt_dir);
	/* 只读接口 */
	const cv::Mat& raw()       const { return raw_; }
	const cv::Mat& demMeters() const { return dem_m_; }
	int width()  const { return raw_.cols; }
	int height() const { return raw_.rows; }
	double minElevationM() const { return min_elev_m_; }
	double maxElevationM() const { return min_elev_m_ + delta_h_m_; }
	double deltaHeightM()  const { return delta_h_m_; }
private:
	cv::Mat raw_;          // 原始单通道 32S/32F
	cv::Mat dem_m_;        // 物理高程 CV_64FC1 单位米
	double min_elev_m_ = 0;
	double delta_h_m_ = 0;
	/* ---------- 内部工具 ---------- */
    /* 读 TIFF */
	void readTiff_(const std::string& path);
	/* 解码到米 */
	void decodeToMeters_();
	/* 导出 txt + 8/16 位预览图 */
	void exportResults_(const std::string& img_dir,const std::string& txt_path) const;

	static cv::Mat normalize01_(const cv::Mat& src);
	static cv::Mat to8U_(const cv::Mat& src);
	static cv::Mat to16U_(const cv::Mat& src);
	void saveImage_(const std::string& path, const cv::Mat& img) const;
};