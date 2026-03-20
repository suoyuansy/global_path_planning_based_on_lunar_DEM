#include "PathPlanning_Global_API.hpp"
#include "Dem.hpp"
#include "TerrainSlopeAspect.hpp"
#include "TerrainRoughness.hpp"
#include "TerrainStepEdge.hpp"
#include "TerrainObstacleExpand.hpp"
#include "TerrainCostmapFusion.hpp"

#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <cmath>
#include <algorithm>

PathPlanning_Global_API::BuildResult
PathPlanning_Global_API::buildFromTiff(const std::string& tiff_path,
    const std::string& output_dir,
    double grid_size) {
    if (grid_size <= 0.0) {
        throw std::runtime_error("grid_size must be > 0.");
    }

    namespace fs = std::filesystem;
    fs::create_directories(output_dir);

    // 只输出 global_path_file/dem.txt
    Dem dem(tiff_path, output_dir, true, true);

    // 中间过程不导出
    TerrainSlopeAspect tsa(dem.demMeters(), output_dir, grid_size, 20.0, 1e10, false);
    TerrainRoughness rough(dem.demMeters(), output_dir, grid_size, 0.15, 1e10, false);
    TerrainStepEdge step(dem.demMeters(), output_dir, 0.4, 1e10, false);
    TerrainObstacleExpand expand(
        tsa.obstacle(),
        rough.obstacle(),
        step.step_obstacle(),
        output_dir,
        1000.0,
        grid_size,
        false
    );
    TerrainCostmapFusion fusion(tsa, rough, step, expand, output_dir, false);

    BuildResult result;
    result.dem = dem.demMeters().clone();
    result.costmap = fusion.costmap_merge_expand().clone();

    saveCostmapTxt(result.costmap, output_dir + "/costmap.txt");
    saveCostmapVis(result.costmap, output_dir + "/costmap_vis.jpg");

    return result;
}

void PathPlanning_Global_API::saveCostmapTxt(const cv::Mat& costmap, const std::string& txt_path) {
    if (costmap.empty() || costmap.type() != CV_64FC1) {
        throw std::runtime_error("saveCostmapTxt: costmap must be CV_64FC1.");
    }

    std::ofstream ofs(txt_path);
    if (!ofs.is_open()) {
        throw std::runtime_error("Cannot open costmap output file: " + txt_path);
    }

    ofs << std::fixed << std::setprecision(6);
    for (int y = 0; y < costmap.rows; ++y) {
        for (int x = 0; x < costmap.cols; ++x) {
            ofs << costmap.at<double>(y, x);
            if (x + 1 < costmap.cols) ofs << " ";
        }
        if (y + 1 < costmap.rows) ofs << "\n";
    }
}

cv::Mat PathPlanning_Global_API::visualizeCostmap_(const cv::Mat& costmap) {
    if (costmap.empty() || costmap.type() != CV_64FC1) {
        throw std::runtime_error("visualizeCostmap_: costmap must be CV_64FC1.");
    }

    cv::Mat out(costmap.rows, costmap.cols, CV_8UC1);

    for (int y = 0; y < costmap.rows; ++y) {
        const double* src = costmap.ptr<double>(y);
        unsigned char* dst = out.ptr<unsigned char>(y);

        for (int x = 0; x < costmap.cols; ++x) {
            double v = src[x];

            if (std::abs(v - 1.0) < 1e-6) {
                dst[x] = 255;  // 障碍
            }
            else {
                v = std::clamp(v, 0.0, 1.0);
                int pix = static_cast<int>(v * 254.0 + 0.5);
                dst[x] = static_cast<unsigned char>(std::clamp(pix, 0, 254));
            }
        }
    }
    return out;
}

void PathPlanning_Global_API::saveCostmapVis(const cv::Mat& costmap, const std::string& img_path) {
    cv::Mat vis = visualizeCostmap_(costmap);
    if (!cv::imwrite(img_path, vis)) {
        throw std::runtime_error("Failed to write costmap image: " + img_path);
    }
}