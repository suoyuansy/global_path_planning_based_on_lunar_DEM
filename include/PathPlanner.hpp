#pragma once
#include <opencv2/core.hpp>
#include <vector>
#include <limits>

/**
 * @brief 路径规划器
 *
 * 当前支持：
 * 1) AStar        : 八方向 A*，适合静态栅格图快速规划
 * 2) DStarLite    : 适合全局时变代价地图的增量重规划
 * 3) HybridAStar  : 适合局部规划，考虑车辆运动学约束
 * 4) Bidirectional: 双向A*规划,用于局部路径调整全局路径点，规划速度更快
 * 注意：
 * - 目前统一对外仍返回 std::vector<cv::Point>，以便兼容你现有工程。
 * - 其中 Hybrid A* 内部实际维护了 (x, y, yaw) 状态，但最终仍投影/离散为点序列返回。
 */

class PathPlanner {
public:
    enum class Method {
        AStar = 0,
        DStarLite,
        HybridAStar,
        BidirectionalAStar
    };

    /**
     * @brief 统一规划入口
     * @param method   规划方法
     * @param costmap  代价地图，要求 CV_64FC1
     * @param start    起点（栅格坐标）
     * @param goal     终点（栅格坐标）
     * @return 路径点序列，若无解则返回空数组
     */

    static std::vector<cv::Point> plan(Method method,const cv::Mat& costmap,const cv::Point& start,const cv::Point& goal);

    /**
     * @brief 八方向 A* 规划
     */
    static std::vector<cv::Point> planAStar(const cv::Mat& costmap,const cv::Point& start,const cv::Point& goal);


    /**
    * @brief D* Lite 规划（适用于全局增量重规划）
    *
    * 说明：
    * - 该函数内部维护静态缓存。
    * - 当“地图尺寸/目标点/参数”不变，仅 costmap 或 start 变化时，
    *   会尝试复用上一次搜索结果并执行增量修复。
    * - 非常适合你后续“光照更新 -> costmap变化 -> 再次调用同一个接口重规划”的场景。
    *
    * @param safe_radius  安全检查半径（单位：栅格）
    * @param cost_weight  代价图权重，g += 步长 + cost_weight * costmap(y,x)
    */
    static std::vector<cv::Point> planDStarLite(
        const cv::Mat& costmap,
        const cv::Point& start,
        const cv::Point& goal,
        int safe_radius = 2,
        double cost_weight = 10.0
    );

    /**
     * @brief Hybrid A* 规划（适用于局部运动学规划）
     *
     * 重要说明：
     * - 对外接口仅暴露地图、起点、终点、起终点航向。
     * - start / goal 仍使用栅格坐标。
     * - 所有运动学和规划相关的长度参数在函数内部先按“米”定义，
     *   再依据地图分辨率换算为“栅格”参与搜索。
     * - 当前地图分辨率固定为 0.1m / cell。
     *
     * @param costmap     代价地图（CV_64F）
     * @param start       起点栅格坐标
     * @param goal        终点栅格坐标
     * @param start_yaw   起点航向（弧度）。若传 NaN，则默认指向 goal
     * @param goal_yaw    终点期望航向（弧度）。若传 NaN，则默认与 start_yaw 相同
     */
    static std::vector<cv::Point> planHybridAStar(
        const cv::Mat& costmap,
        const cv::Point& start,
        const cv::Point& goal,
        double start_yaw = std::numeric_limits<double>::quiet_NaN(),
        double goal_yaw = std::numeric_limits<double>::quiet_NaN()
    );

    // 添加双向A*的声明
    /**
     * @brief 双向A*规划（Bidirectional A*）
     *
     * 特点：
     * - 同时从起点和终点进行搜索
     * - 使用两套g值、f值和父节点指针
     * - 当两个搜索前沿相遇时停止
     * - 相比普通A*，搜索节点数减少约50%，速度更快
     *
     * @param costmap      代价地图
     * @param start        起点
     * @param goal         终点
     * @param safe_radius  安全检查半径
     * @param cost_weight  代价图权重
     * @return 路径点序列
     */
    static std::vector<cv::Point> planBidirectionalAStar(
        const cv::Mat& costmap,
        const cv::Point& start,
        const cv::Point& goal,
        int safe_radius = 2,
        double cost_weight = 10.0
    );


private:
    static bool isObstacle_(const cv::Mat& costmap, int x, int y);
    static bool isSafe_(const cv::Mat& costmap, int x, int y, int safe_radius = 2);
};