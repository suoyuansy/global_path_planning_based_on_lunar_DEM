//#include "Dem.hpp"
//#include <sstream>
//#include <stdexcept>
//
//
//void Dem::setRaw(const cv::Mat& raw) {
//    if (raw.empty()) {
//        throw std::runtime_error("Dem::setRaw: raw is empty.");
//    }
//    if (raw.channels() != 1) {
//        throw std::runtime_error("Dem::setRaw: raw must be single-channel.");
//    }
//    raw_ = raw.clone();
//    cv_type_ = raw_.type();
//}
//
//const cv::Mat& Dem::raw() const { return raw_; }
//
//void Dem::setDemMeters(const cv::Mat& dem_m) {
//    if (dem_m.empty()) {
//        throw std::runtime_error("Dem::setDemMeters: dem is empty.");
//    }
//    if (dem_m.type() != CV_64FC1 || dem_m.channels() != 1) {
//        throw std::runtime_error("Dem::setDemMeters: dem must be CV_64FC1 single-channel.");
//    }
//    dem_m_ = dem_m.clone();
//}
//
//const cv::Mat& Dem::demMeters() const { return dem_m_; }
//
//
//void Dem::setElevationEncoding(double min_elev_m, double delta_h_m) {
//    if (delta_h_m <= 0.0) {
//        throw std::runtime_error("Dem::setElevationEncoding: delta_h_m must be > 0.");
//    }
//    min_elev_m_ = min_elev_m;
//    delta_h_m_ = delta_h_m;
//    has_encoding_ = true;
//}
//
//bool Dem::hasElevationEncoding() const {
//    return has_encoding_;
//}
//
//double Dem::minElevationM() const {
//    return min_elev_m_;
//}
//
//double Dem::maxElevationM() const {
//    return min_elev_m_ + delta_h_m_;
//}
//
//double Dem::deltaHeightM() const {
//    return delta_h_m_;
//}
//
//int Dem::width() const { return raw_.cols; }
//int Dem::height() const { return raw_.rows; }
//int Dem::cvType() const { return cv_type_; }
//
//std::string Dem::summary() const {
//    std::ostringstream oss;
//    oss << "DEM Summary\n";
//    oss << "Size: " << width() << " x " << height() << "\n";
//    oss << "OpenCV raw type: " << cv_type_ << "\n";
//
//    if (has_encoding_) {
//        oss << "Min elevation (m): " << min_elev_m_ << "\n";
//        oss << "Max elevation (m): " << maxElevationM() << "\n";
//        oss << "Height range ¦¤H (m): " << delta_h_m_ << "\n";
//    }
//    else {
//        oss << "Elevation encoding: (not set)\n";
//    }
//
//    oss << "DEM meters ready: " << (!dem_m_.empty() ? "YES" : "NO") << "\n";
//    return oss.str();
//}
