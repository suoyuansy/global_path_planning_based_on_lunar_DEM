#include "Dem.hpp"
#include "TerrainSlopeAspect.hpp"
#include "TerrainRoughness.hpp"
#include "TerrainStepEdge.hpp"
#include "TerrainObstacleExpand.hpp"
#include "TerrainCostmapFusion.hpp"
#include "PathPlanningInteractive.hpp"
#include "PathPlanning_Local_API.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>

#include <iostream>
#include <filesystem>



int main(int argc, char** argv) {
    // 关闭 OpenCV 冗余日志
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

    try {

        // =========================================================
       // 模式1：Python / 局部路径规划调用模式
       //
       // 用法：
       // program.exe dem.txt start_x start_y goal_x goal_y grid_size output.txt
       //
       // 参数说明：
       // argv[1] : dem_txt_path
       // argv[2] : start_x
       // argv[3] : start_y
       // argv[4] : goal_x
       // argv[5] : goal_y
       // argv[6] : grid_size
       // argv[7] : output_txt_path，是一个文件夹路径，路径的txt与costmap的txt文件都生成在这个文件夹下面
       // =========================================================
        if (argc == 8) {//说明是通过命令行或 Python 传了 7 个参数
            const std::string dem_txt_path = argv[1];
            const int start_x = std::stoi(argv[2]);
            const int start_y = std::stoi(argv[3]);
            const int goal_x = std::stoi(argv[4]);
            const int goal_y = std::stoi(argv[5]);
            const double grid_size = std::stod(argv[6]);
            const std::string output_txt_path = argv[7];

            // 从 txt 读取 DEM
            cv::Mat dem = PathPlanning_Local_API::loadDEMFromTxt(dem_txt_path);

            // 调用 API 完成规划
            PathPlanning_Local_API::PlanResult result = PathPlanning_Local_API::planFromDEM(dem,cv::Point(start_x, start_y),cv::Point(goal_x, goal_y),grid_size);

            // 将结果保存到路径
            PathPlanning_Local_API::saveResultToFile(result, output_txt_path);

            return 0;
        }

        // =========================================================
       // 模式2：外部直接传入 costmap 的局部路径规划模式
       //
       // 用法：
       // program.exe costmap.txt start_x start_y goal_x goal_y output_dir
       //
       // 参数说明：
       // argv[1] : costmap_txt_path
       // argv[2] : start_x
       // argv[3] : start_y
       // argv[4] : goal_x
       // argv[5] : goal_y
       // argv[6] : output_dir
       // =========================================================
        if (argc == 7) {
            const std::string costmap_txt_path = argv[1];
            const int start_x = std::stoi(argv[2]);
            const int start_y = std::stoi(argv[3]);
            const int goal_x = std::stoi(argv[4]);
            const int goal_y = std::stoi(argv[5]);
            const std::string output_txt_path = argv[6];

            cv::Mat costmap = PathPlanning_Local_API::loadCostmapFromTxt(costmap_txt_path);

            PathPlanning_Local_API::PlanResult result =PathPlanning_Local_API::planFromCostmap(costmap,cv::Point(start_x, start_y),cv::Point(goal_x, goal_y));

            PathPlanning_Local_API::saveResultToFile(result, output_txt_path);

            return 0;
        }

        // =========================================================
        // 模式3：外部调用交互模式
        //
        // 用法：
        // program.exe <tiff_path> <tiff_color_path> <result_dir>
        //
        // 参数说明：
        // argv[1] : tiff_path
        // argv[2] : tiff_coler_path
        // argv[3] : result_file
        //
        // 说明：
        // 该模式与原始交互模式流程一致，
        // 只是把 tif、彩色底图、输出目录改为外部传入，
        // 方便 Python 或其他程序调用。
        // =========================================================
        if (argc == 4) {

            const std::string tiff_path = argv[1];
            const std::string tiff_coler_path = argv[2];
            const std::string result_file = argv[3];

            namespace fs = std::filesystem;
            fs::create_directories(result_file);   // 保证根目录存在

            std::cout << "\n******** Part 1: Loading DEM, converting to meters and exporting raw data ********\n" << std::endl;
            Dem dem(tiff_path, result_file);

            std::cout << "\n******** Part 2: Computing slope, aspect and terrain costs, then exporting results ********\n" << std::endl;
            TerrainSlopeAspect tsa(dem.demMeters(), result_file);

            std::cout << "\n******** Part 3: Computing rover-scale roughness and cost, then exporting results ********\n" << std::endl;
            TerrainRoughness rough(dem.demMeters(), result_file);

            std::cout << "\n******** Part 4: Computing step-edge and cost, then exporting results ********\n" << std::endl;
            TerrainStepEdge step(dem.demMeters(), result_file);

            std::cout << "\n******** Part 5: Expanding obstacle masks for safe planning ********\n" << std::endl;
            TerrainObstacleExpand expand(tsa.obstacle(),rough.obstacle(),step.step_obstacle(),result_file);

            std::cout << "\n******** Part 6: Fusing 16 costmaps for all planning strategies, then exporting results ********\n" << std::endl;
            TerrainCostmapFusion fusion(tsa, rough, step, expand, result_file);

            std::cout << "\n******** Part 7: Interactive path planning ********\n" << std::endl;
            PathPlanningInteractive planner(tiff_coler_path);

            return 0;
        }

        // =========================================================
        // 模式4：原始直接运行模式
        //
        // 不带参数运行时，继续执行原来的：
        // 1. 读取 tif
        // 2. 计算坡度 / 粗糙度 / 台阶边缘
        // 3. 障碍膨胀
        // 4. 代价图融合
        // 5. 进入交互式路径规划
        // =========================================================
        if (argc == 1) {//说明是直接在 VS 里点“本地调试器运行”或 F5，没有传命令行参数，所以会进入
            const std::string tiff_path = "data/CE7DEM_1km.tif";
            const std::string tiff_coler_path = "data/CE7DEM_1km_color.png";
            const std::string result_file = "out_put";

            namespace fs = std::filesystem;
            fs::create_directories(result_file);   // 保证根目录存在

            //std::cout << "\n******** Part 1:Loading DEM, converting to meters and exporting raw data ********\n" << std::endl;
            //Dem dem(tiff_path, result_file);

            //std::cout << "\n******** Part 2:Computing slope, aspect and terrain costs, then exporting results ********\n" << std::endl;
            //TerrainSlopeAspect tsa(dem.demMeters(), result_file);

            //std::cout << "\n******** Part 3:Computing rover-scale roughness and cost, then exporting results ********\n" << std::endl;
            //TerrainRoughness rough(dem.demMeters(), result_file);

            //std::cout << "\n******** Part 4:Computing step-edge and cost, then exporting results ********\n" << std::endl;;
            //TerrainStepEdge step(dem.demMeters(), result_file);

            //std::cout << "\n******** Part 5:Expanding obstacle masks for safe planning ********\n" << std::endl;;
            //TerrainObstacleExpand expand(tsa.obstacle(), rough.obstacle(), step.step_obstacle(), result_file);

            //std::cout << "\n******** Part 6:Fusing 16 costmaps for all planning strategies, then exporting results ********\n" << std::endl;;
            //TerrainCostmapFusion fusion(tsa, rough, step, expand, result_file);

            std::cout << "\n******** Part 7:Interactive path planning ********\n" << std::endl;
            PathPlanningInteractive planner(tiff_coler_path);

            return 0;
        }
        // =========================================================
              // 其他参数数量都视为输入错误
              //
              // 当前支持四种模式：
              // 1. 外部调用局部路径规划模式（从 DEM 计算 costmap，再规划）
              //    对应 argc == 8
              // 2. 外部直接传入 costmap 的局部路径规划模式
              //    对应 argc == 7
              // 3. 外部调用全局路径规划交互模式
              //    对应 argc == 4
              // 4. 原始直接运行模式（无参数）
              //    对应 argc == 1
              // =========================================================
        std::cerr
            << "Usage:\n\n"

            << "  Mode 1) External local path planning from DEM:\n"
            << "     " << argv[0]
            << " <dem_txt_path> <start_x> <start_y> <goal_x> <goal_y> <grid_size> <output_dir>\n"
            << "     Example:\n"
            << "     " << argv[0]
            << " dem.txt 37 52 966 966 1.0 local_planning_output\n\n"

            << "  Mode 2) External local path planning from costmap:\n"
            << "     " << argv[0]
            << " <costmap_txt_path> <start_x> <start_y> <goal_x> <goal_y> <output_dir>\n"
            << "     Example:\n"
            << "     " << argv[0]
            << " costmap.txt 37 52 966 966 local_planning_output\n\n"

            << "  Mode 3) External global path planning interactive mode:\n"
            << "     " << argv[0]
            << " <tiff_path> <tiff_color_path> <result_dir>\n"
            << "     Example:\n"
            << "     " << argv[0]
            << " data/CE7DEM_1km.tif data/CE7DEM_1km_color.png out_put\n\n"

            << "  Mode 4) Default interactive mode:\n"
            << "     " << argv[0] << "\n"
            << "     (Run without any arguments)\n";

        return 1;
    }
    catch (const std::exception& e) {
        std::cerr << "[Error] " << e.what() << "\n";
        return 1;
    }
}