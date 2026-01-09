#include "TiffReader.hpp"
#include "DemBuilder.hpp"
#include "DemIO.hpp"
#include "TerrainDerivatives.hpp"
#include "TerrainCost.hpp"
#include "TerrainIO.hpp"

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

        // 确保输出目录存在（避免保存失败）
        std::filesystem::create_directories(out_image_dir);
        std::filesystem::create_directories(out_txt_dir);

        std::cout << "******** Part 1:Loading DEM, converting to meters and exporting raw data ********\n"<<std::endl;
        //  读 TIFF 到 dem.raw()
        Dem dem;
        TiffReader::ReadSingleChannel32Bit(tiff_path, dem);
        std::cout << dem.summary() << "\n";

        //  根据 raw 类型决定是否需要手动输入
        const int depth = dem.raw().depth();

        if (depth == CV_32S) {
            double min_elev_m = 0.0;
            double max_elev_m = 0.0;

            std::cout << "Raw is CV_32S (encoded). Please enter elevation range.\n";
            std::cout << "Enter MIN elevation (m): ";
            std::cin >> min_elev_m;
            std::cout << "Enter MAX elevation (m): ";
            std::cin >> max_elev_m;

            if (max_elev_m <= min_elev_m) {
                throw std::runtime_error("Max elevation must be greater than min elevation.");
            }

            double delta_h_m = max_elev_m - min_elev_m;
            DemBuilder::BuildMetersFromUInt32Encoding(dem, min_elev_m, delta_h_m);
        }
        else if (depth == CV_32F) {
            std::cout << "Raw is CV_32F (assumed already meters). No manual range needed.\n";
            DemBuilder::BuildMetersFromUInt32Encoding(dem, 0.0, 0.0); // 走CV_32F分支，参数不使用
        }
        else {
            throw std::runtime_error("Unexpected depth (should be CV_32S or CV_32F).");
        }

        std::cout << "\n" << dem.summary() << "\n";

        //  输出 DEM 到 txt + 输出 raw 预览图
        DemIO::ExportDemToText(dem, out_txt_dir+"/dem.txt");
        DemIO::ExportAndPreview(dem, out_image_dir);

        std::cout << "\nOutputs:\n";
        std::cout << "  " << out_txt_dir << "/dem.txt\n";
        std::cout << "  " << out_image_dir << "/raw_8u.png\n";
        std::cout << "  " << out_image_dir << "/raw_16u.png\n";

        std::cout << "******** Part 2:Computing slope, aspect and terrain costs, then exporting results ********\n" << std::endl;
         //  计算坡度/坡向
         // 栅格尺寸 g：CE7DEM_1km => 1000m；如果你的 DEM 是 1m，就改成 1.0
        const double g = 1.0;
        auto sa = TerrainDerivatives::ComputeSlopeAspect_3rdOrder(dem.demMeters(), g);

        // 输出坡度/坡向栅格文件
        TerrainIO::ExportMat64ToText(sa.slope_deg, out_txt_dir + "/slope_deg.txt", 3);
        TerrainIO::ExportMat64ToText(sa.aspect_deg, out_txt_dir + "/aspect_deg.txt", 3);

        // 代价/障碍（两种都算，图也两种都输出）
        const double theta_max = 20.0;
        const double inf_cost = 1e10;

        auto cost_dist = TerrainCost::BuildCostFromSlope(sa.slope_deg, PlanningStrategy::DistanceShortest, theta_max, inf_cost);
        auto cost_slope = TerrainCost::BuildCostFromSlope(sa.slope_deg, PlanningStrategy::SlopeCostMin, theta_max, inf_cost);

        // 输出障碍/代价矩阵为 txt
        TerrainIO::ExportMat64ToText(cost_dist.cost, out_txt_dir + "/cost_distance.txt", 3);
        TerrainIO::ExportMat64ToText(cost_slope.cost, out_txt_dir + "/cost_slope.txt", 3);

        // 可视化输出
        // 坡度图（0~20 -> 灰度；>=20 255）
        cv::Mat slope_img = TerrainIO::VisualizeSlope8U(sa.slope_deg, theta_max);
        TerrainIO::SaveGrayPng(out_image_dir + "/slope_deg.png", slope_img);

        // 障碍/代价图（两种策略都输出）
        cv::Mat cost_dist_img = TerrainIO::VisualizeCost8U(cost_dist.cost, cost_dist.obstacle, inf_cost);
        cv::Mat cost_slope_img = TerrainIO::VisualizeCost8U(cost_slope.cost, cost_slope.obstacle, inf_cost);

        TerrainIO::SaveGrayPng(out_image_dir + "/cost_distance.png", cost_dist_img);
        TerrainIO::SaveGrayPng(out_image_dir + "/cost_slope.png", cost_slope_img);

        std::cout << "\nOutputs:\n";
        std::cout << "  " << out_txt_dir << "/slope_deg.txt\n";
        std::cout << "  " << out_txt_dir << "/aspect_deg.txt\n";
        std::cout << "  " << out_txt_dir << "/cost_distance.txt\n";
        std::cout << "  " << out_txt_dir << "/cost_slope.txt\n";
        std::cout << "  " << out_image_dir << "/slope_deg.png\n";
        std::cout << "  " << out_image_dir << "/cost_distance.png\n";
        std::cout << "  " << out_image_dir << "/cost_slope.png\n";


        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "[Error] " << e.what() << "\n";
        return 1;
    }
}