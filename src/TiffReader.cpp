#include "TiffReader.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <stdexcept>

void TiffReader::ReadSingleChannel32Bit(const std::string& tiff_path, Dem& dem) {
    cv::Mat img = cv::imread(tiff_path, cv::IMREAD_UNCHANGED);
    if (img.empty()) {
        throw std::runtime_error("TiffReader: failed to read: " + tiff_path);
    }

    // 如果不是单通道，就拆通道只取第一个（你原先逻辑保留）
    if (img.channels() != 1) {
        std::vector<cv::Mat> ch;
        cv::split(img, ch);
        img = ch[0];
    }

    // 只接受 32-bit：CV_32S / CV_32F
    if (img.depth() != CV_32S && img.depth() != CV_32F) {
        throw std::runtime_error("TiffReader: image is not 32-bit depth (expected CV_32S or CV_32F).");
    }

    dem.setRaw(img);
}