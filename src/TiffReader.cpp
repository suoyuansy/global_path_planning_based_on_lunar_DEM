#include "TiffReader.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <stdexcept>

Dem TiffReader::ReadSingleChannel32Bit(const std::string& tiff_path) {
    cv::Mat img = cv::imread(tiff_path, cv::IMREAD_UNCHANGED);
    if (img.empty()) {
        throw std::runtime_error("TiffReader: failed to read: " + tiff_path);
    }

    if (img.channels() != 1) {//如果读进来的 TIFF 不是单通道，就强行拆通道，只保留第一个通道
        std::vector<cv::Mat> ch;
        cv::split(img, ch);
        img = ch[0];
    }

    // 要求“32位深度”：
    if (img.depth() != CV_32S && img.depth() != CV_32F) {
        // 实际上只有 CV_32S / CV_32F 常见
        // 如果读出来不是 32 位，直接报错，强制发现问题
        throw std::runtime_error("TiffReader: image is not 32-bit depth (expected CV_32S or CV_32F).");
    }

    Dem dem;
    dem.setRaw(img);
    return dem;
}
