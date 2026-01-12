#include "Dem.hpp"
#include "TerrainSlopeAspect.hpp"

#include <iostream>
#include <stdexcept>
#include <filesystem>
#include <opencv2/core/utils/logger.hpp>

int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

    try {
        const std::string tiff_path = "data/CE7DEM_1km.tif";
        const std::string out_image_dir = "out_image";
        const std::string out_txt_dir = "out_txt_file";

        std::cout << "******** Part 1:Loading DEM, converting to meters and exporting raw data ********\n" << std::endl;
        Dem dem(tiff_path, out_image_dir, out_txt_dir);

        std::cout << "\n******** Part 2:Computing slope, aspect and terrain costs, then exporting results ********\n" << std::endl;
        TerrainSlopeAspect tsa(dem.demMeters(), out_image_dir, out_txt_dir);

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "[Error] " << e.what() << "\n";
        return 1;
    }
}