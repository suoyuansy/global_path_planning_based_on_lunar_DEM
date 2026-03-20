#include "PathPlanning_Local_API.hpp"

#include "TerrainSlopeAspect.hpp"
#include "TerrainRoughness.hpp"
#include "TerrainStepEdge.hpp"
#include "TerrainObstacleExpand.hpp"
#include "TerrainCostmapFusion.hpp"
#include "PathPlanner.hpp"

#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <cmath>
#include <sstream>
#include <vector>
#include <string>

bool PathPlanning_Local_API::isObstacle_(const cv::Mat& costmap, int x, int y)
{
    return std::abs(costmap.at<double>(y, x) - 1.0) < 1e-6;
}


cv::Mat PathPlanning_Local_API::loadDEMFromTxt(const std::string& txt_path)
{
    std::ifstream ifs(txt_path);
    if (!ifs.is_open()) {
        throw std::runtime_error("Cannot open DEM txt file: " + txt_path);
    }

    std::vector<std::vector<double>> data;
    std::string line;
    size_t cols = 0;  // 列数由第一行有效数据决定

    // 按行读取 txt
    while (std::getline(ifs, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::vector<double> row;
        double value;

        // 读取一整行中的所有高程值
        while (ss >> value) {
            row.push_back(value);
        }

        // 空行或无有效数字则跳过
        if (row.empty()) continue;

        // 第一行有效数据确定列数
        if (cols == 0) {
            cols = row.size();
        }
        // 之后每一行都必须与第一行列数一致
        else if (row.size() != cols) {
            throw std::runtime_error("DEM txt file is not rectangular: " + txt_path);
        }

        data.push_back(std::move(row));
    }

    // 检查是否成功读到数据
    if (data.empty() || cols == 0) {
        throw std::runtime_error("DEM txt file is empty or invalid: " + txt_path);
    }

    const int rows = static_cast<int>(data.size());
    const int cols_int = static_cast<int>(cols);

    // 创建 CV_64FC1 类型 DEM 矩阵
    cv::Mat dem(rows, cols_int, CV_64FC1);

    // 按行写入矩阵
    // data[y][x] -> dem.at<double>(y, x)
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols_int; ++x) {
            dem.at<double>(y, x) = data[y][x];
        }
    }

    return dem;
}

cv::Mat PathPlanning_Local_API::loadCostmapFromTxt(const std::string& txt_path)
{
    std::ifstream ifs(txt_path);
    if (!ifs.is_open()) {
        throw std::runtime_error("Cannot open costmap txt file: " + txt_path);
    }

    std::vector<std::vector<double>> data;
    std::string line;
    size_t cols = 0;

    while (std::getline(ifs, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::vector<double> row;
        double value;

        while (ss >> value) {
            row.push_back(value);
        }

        if (row.empty()) continue;

        if (cols == 0) {
            cols = row.size();
        }
        else if (row.size() != cols) {
            throw std::runtime_error("Costmap txt file is not rectangular: " + txt_path);
        }

        data.push_back(std::move(row));
    }

    if (data.empty() || cols == 0) {
        throw std::runtime_error("Costmap txt file is empty or invalid: " + txt_path);
    }

    const int rows = static_cast<int>(data.size());
    const int cols_int = static_cast<int>(cols);

    cv::Mat costmap(rows, cols_int, CV_64FC1);

    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols_int; ++x) {
            costmap.at<double>(y, x) = data[y][x];
        }
    }

    return costmap;
}



PathPlanning_Local_API::PlanResult PathPlanning_Local_API::planFromDEM(const cv::Mat& dem,const cv::Point& start,const cv::Point& goal, const double grid_size)
{
    if (dem.empty() || dem.type() != CV_64FC1)
        throw std::runtime_error("planFromDEM: dem must be CV_64FC1.");
    if (grid_size <= 0.0)
        throw std::runtime_error("planFromDEM: grid_size must be > 0.");

    if (start.x < 0 || start.x >= dem.cols || start.y < 0 || start.y >= dem.rows)
        throw std::runtime_error("planFromDEM: start is out of bounds.");
    if (goal.x < 0 || goal.x >= dem.cols || goal.y < 0 || goal.y >= dem.rows)
        throw std::runtime_error("planFromDEM: goal is out of bounds.");

    const std::string dummy_root;

    TerrainSlopeAspect tsa(dem, dummy_root, grid_size, 20.0, 1e10, false);
    TerrainRoughness rough(dem, dummy_root, grid_size, 0.15, 1e10, false);
    TerrainStepEdge step(dem, dummy_root, 0.4, 1e10, false);

    TerrainObstacleExpand expand( tsa.obstacle(),rough.obstacle(),step.step_obstacle(),dummy_root, 1000 ,grid_size, false);

    TerrainCostmapFusion fusion(tsa,rough,step,expand,dummy_root,false);

    const cv::Mat& costmap = fusion.costmap_merge_expand();

    PlanResult result;
    result.costmap = costmap.clone();   // 无论是否有路径，都保存 costmap

    const bool start_is_obstacle = isObstacle_(costmap, start.x, start.y);
    const bool goal_is_obstacle = isObstacle_(costmap, goal.x, goal.y);
    
    if (start_is_obstacle && goal_is_obstacle) {
        result.status = PlanStatus::START_AND_GOAL_ARE_OBSTACLES;
        return result;
    }

    if (start_is_obstacle) {
        result.status = PlanStatus::START_IS_OBSTACLE;
        return result;
    }

    if (goal_is_obstacle) {
        result.status = PlanStatus::GOAL_IS_OBSTACLE;
        return result;
    }

    // 再做路径规划
    result.path = PathPlanner::plan(PathPlanner::Method::AStar,costmap,start,goal);

    // 空路径表示未找到路径
    if (result.path.empty()) {
        result.status = PlanStatus::NO_PATH_FOUND;
    }
    else {
        result.status = PlanStatus::OK;
    }

    return result;
}


PathPlanning_Local_API::PlanResult PathPlanning_Local_API::planFromCostmap(const cv::Mat& costmap,const cv::Point& start,const cv::Point& goal)
{
    if (costmap.empty() || costmap.type() != CV_64FC1)
        throw std::runtime_error("planFromCostmap: costmap must be CV_64FC1.");

    if (start.x < 0 || start.x >= costmap.cols || start.y < 0 || start.y >= costmap.rows)
        throw std::runtime_error("planFromCostmap: start is out of bounds.");

    if (goal.x < 0 || goal.x >= costmap.cols || goal.y < 0 || goal.y >= costmap.rows)
        throw std::runtime_error("planFromCostmap: goal is out of bounds.");

    PlanResult result;
    result.costmap = costmap.clone();

    const bool start_is_obstacle = isObstacle_(costmap, start.x, start.y);
    const bool goal_is_obstacle = isObstacle_(costmap, goal.x, goal.y);

    if (start_is_obstacle && goal_is_obstacle) {
        result.status = PlanStatus::START_AND_GOAL_ARE_OBSTACLES;
        return result;
    }

    if (start_is_obstacle) {
        result.status = PlanStatus::START_IS_OBSTACLE;
        return result;
    }

    if (goal_is_obstacle) {
        result.status = PlanStatus::GOAL_IS_OBSTACLE;
        return result;
    }

    result.path = PathPlanner::plan(PathPlanner::Method::AStar, costmap, start, goal);

    if (result.path.empty()) {
        result.status = PlanStatus::NO_PATH_FOUND;
    }
    else {
        result.status = PlanStatus::OK;
    }

    return result;
}


void PathPlanning_Local_API::saveResultToFile(const PlanResult& result,const std::string& output_path, bool Isexportcostmap)
{
    namespace fs = std::filesystem;

    fs::path output_dir(output_path);

    // 确保输出目录存在
    fs::create_directories(output_dir);

    // 生成两个文件路径
    fs::path path_file = output_dir / "path.txt";
    fs::path costmap_file = output_dir / "costmap.txt";

    // ============================
    // 1. 写入路径结果
    // ============================

    std::ofstream ofs(path_file.string());
    if (!ofs.is_open()) {
        throw std::runtime_error("Cannot open path output file: " + path_file.string());
    }

    switch (result.status) {
    case PlanStatus::START_IS_OBSTACLE:
        ofs << "START_IS_OBSTACLE";
        break;

    case PlanStatus::GOAL_IS_OBSTACLE:
        ofs << "GOAL_IS_OBSTACLE";
        break;

    case PlanStatus::START_AND_GOAL_ARE_OBSTACLES:
        ofs << "START_AND_GOAL_ARE_OBSTACLES";
        break;

    case PlanStatus::NO_PATH_FOUND:
        ofs << "NO_PATH_FOUND";
        break;

    case PlanStatus::OK:
        for (size_t i = 0; i < result.path.size(); ++i) {
            ofs << "(" << result.path[i].x << "," << result.path[i].y << ")";
            if (i + 1 < result.path.size()) {
                ofs << "->";
            }
        }
        break;

    default:
        ofs << "UNKNOWN_ERROR";
        break;
    }

    ofs.close();

    // ============================
    // 2. 写入 costmap
    // ============================
    if (Isexportcostmap)
    {
        if (result.costmap.empty()) {
            throw std::runtime_error("saveResultToFile: result.costmap is empty.");
        }

        std::ofstream cost_ofs(costmap_file.string());
        if (!cost_ofs.is_open()) {
            throw std::runtime_error("Cannot open costmap output file: " + costmap_file.string());
        }

        for (int y = 0; y < result.costmap.rows; ++y) {
            for (int x = 0; x < result.costmap.cols; ++x) {
                cost_ofs << result.costmap.at<double>(y, x);
                if (x + 1 < result.costmap.cols) {
                    cost_ofs << " ";
                }
            }
            if (y + 1 < result.costmap.rows) {
                cost_ofs << "\n";
            }
        }

        cost_ofs.close();
    }
}