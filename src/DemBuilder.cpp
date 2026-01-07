#include "DemBuilder.hpp"
#include <opencv2/core.hpp>
#include <stdexcept>
#include <cstdint>

void DemBuilder::BuildMetersFromUInt32Encoding(Dem& dem,double min_elev_m,double delta_h_m)
{
    dem.setElevationEncoding(min_elev_m, delta_h_m);

    const cv::Mat& raw = dem.raw();
    if (raw.empty() || raw.channels() != 1) {
        throw std::runtime_error("Invalid raw DEM.");
    }

    constexpr double INV_2_POW_32 = 1.0 / 4294967296.0;
    const double scale = delta_h_m * INV_2_POW_32;

    cv::Mat out(raw.rows, raw.cols, CV_64FC1);

    if (raw.depth() == CV_32S) {//每个像素由一个32位有符号整数（即int）组成
        for (int y = 0; y < raw.rows; ++y) {
            const int32_t* src = raw.ptr<int32_t>(y);
            double* dst = out.ptr<double>(y);
            for (int x = 0; x < raw.cols; ++x) {
                uint32_t u = static_cast<uint32_t>(src[x]);
                dst[x] = static_cast<double>(u) * scale + min_elev_m;
            }
        }
    }
    else if (raw.depth() == CV_32F) {//
        for (int y = 0; y < raw.rows; ++y) {
            const float* src = raw.ptr<float>(y);
            double* dst = out.ptr<double>(y);
            for (int x = 0; x < raw.cols; ++x) {
                double v = std::max(0.0, static_cast<double>(src[x]));
                dst[x] = v * scale + min_elev_m;
            }
        }
    }
    else {
        throw std::runtime_error("Unsupported raw depth.");
    }

    dem.setDemMeters(out);
}
