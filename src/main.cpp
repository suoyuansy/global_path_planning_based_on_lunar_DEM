#include "TiffReader.hpp"
#include "DemBuilder.hpp"
#include "DemIO.hpp"

#include <iostream>
#include <opencv2/core/utils/logger.hpp>

int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

    try { 
        const std::string tiff_path = "data/CE7DEM_1km.tif";
        const std::string out_dir = "out";

        // 读 TIFF（32-bit raw 编码）
        Dem dem = TiffReader::ReadSingleChannel32Bit(tiff_path);
        std::cout << dem.summary() << "\n";

        //  手动输入物理高程范围
        double min_elev_m = 0.0;
        double max_elev_m = 0.0;

        std::cout << "Enter MIN elevation (m): ";
        std::cin >> min_elev_m;

        std::cout << "Enter MAX elevation (m): ";
        std::cin >> max_elev_m;

        if (max_elev_m <= min_elev_m) {
            throw std::runtime_error("Max elevation must be greater than min elevation.");
        }

        double delta_h_m = max_elev_m - min_elev_m;

        // 按 2^32 编码生成 DEM（CV_64F, meters）
        DemBuilder::BuildMetersFromUInt32Encoding(
            dem,
            min_elev_m,
            delta_h_m
        );


        std::cout << "\n"<< dem.summary() << "\n";

        // 仅导出\显示 raw
        DemIO::ExportAndPreview(dem, out_dir);

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "[Error] " << e.what() << "\n";
        return 1;
    }
}
