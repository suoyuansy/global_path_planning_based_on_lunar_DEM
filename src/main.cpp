#include "Dem.hpp"
#include "TerrainSlopeAspect.hpp"
#include "TerrainRoughness.hpp"

#include <iostream>
#include <stdexcept>
#include <filesystem>
#include <opencv2/core/utils/logger.hpp>

int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

    try {
        const std::string tiff_path = "data/CE7DEM_1km.tif";

        const std::string result_file = "out_put";
        namespace fs = std::filesystem;
        fs::create_directories(result_file);   // 保证根目录存在

        std::cout << "\n******** Part 1:Loading DEM, converting to meters and exporting raw data ********\n" << std::endl;
        Dem dem(tiff_path, result_file);

        std::cout << "\n******** Part 2:Computing slope, aspect and terrain costs, then exporting results ********\n" << std::endl;
        TerrainSlopeAspect tsa(dem.demMeters(), result_file);

        std::cout << "\n******** Part 3:Computing rover-scale roughness & cost, then exporting results ********\n" << std::endl;
        TerrainRoughness rough(dem.demMeters(), result_file);

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "[Error] " << e.what() << "\n";
        return 1;
    }
}