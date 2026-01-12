//#include "DemBuilder.hpp"
//#include <opencv2/core.hpp>
//#include <stdexcept>
//#include <cstdint>
//
//
//void DemBuilder::BuildMetersFromUInt32Encoding(Dem& dem, double min_elev_m, double delta_h_m) {
//    const cv::Mat& raw = dem.raw();
//    if (raw.empty() || raw.channels() != 1) {
//        throw std::runtime_error("Invalid raw DEM.");
//    }
//
//    cv::Mat out(raw.rows, raw.cols, CV_64FC1);
//
//    if (raw.depth() == CV_32S) {
//        // 32-bit 整型：按 2^32 编码解码，需要 min + delta
//        if (delta_h_m <= 0.0) {
//            throw std::runtime_error("delta_h_m must be > 0 for CV_32S decoding.");
//        }
//
//        dem.setElevationEncoding(min_elev_m, delta_h_m);
//        const double scale = delta_h_m / 4294967296.0; // 2^32
//
//        for (int y = 0; y < raw.rows; ++y) {
//            const int32_t* src = raw.ptr<int32_t>(y);
//            double* dst = out.ptr<double>(y);
//            for (int x = 0; x < raw.cols; ++x) {
//                uint32_t u = static_cast<uint32_t>(src[x]); // 保留bit模式按uint32解释
//                dst[x] = static_cast<double>(u) * scale + min_elev_m;
//            }
//        }
//    }
//    else if (raw.depth() == CV_32F) {
//        // 32-bit float：认为已经是米单位高程（直接转 double）
//        // 不设置 encoding（因为不需要 min/max）
//        for (int y = 0; y < raw.rows; ++y) {
//            const float* src = raw.ptr<float>(y);
//            double* dst = out.ptr<double>(y);
//            for (int x = 0; x < raw.cols; ++x) {
//                dst[x] = static_cast<double>(src[x]);
//            }
//        }
//    }
//    else {
//        throw std::runtime_error("Unsupported raw depth.");
//    }
//
//    dem.setDemMeters(out);
//}