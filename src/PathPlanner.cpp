#include "PathPlanner.hpp"
#include <queue>
#include <tuple>
#include <cmath>
#include <algorithm>
#include <stdexcept>

bool PathPlanner::isObstacle_(const cv::Mat& costmap, int x, int y)
{
    return std::abs(costmap.at<double>(y, x) - 1.0) < 1e-6;
}

bool PathPlanner::isSafe_(const cv::Mat& costmap, int x, int y, int safe_radius)
{
    const int w = costmap.cols;
    const int h = costmap.rows;
    /* 安全区检查 */
    for (int dy = -safe_radius; dy <= safe_radius; ++dy) {
        for (int dx = -safe_radius; dx <= safe_radius; ++dx) {
            int nx = x + dx;
            int ny = y + dy;
            if (nx < 0 || ny < 0 || nx >= w || ny >= h) return false;// 越界也算不安全
            if (isObstacle_(costmap, nx, ny)) return false;
        }
    }
    return true;
}

std::vector<cv::Point> PathPlanner::plan(Method method,const cv::Mat& costmap,const cv::Point& start,const cv::Point& goal)
{
    switch (method) {
    case Method::AStar:
        return planAStar(costmap, start, goal);
    default:
        throw std::runtime_error("Unsupported planning method.");
    }
}

/* ---------------- 八方向 A* 规划（欧氏距离） ---------------- */
std::vector<cv::Point> PathPlanner::planAStar(const cv::Mat& costmap,const cv::Point& start,const cv::Point& goal)
{
    if (costmap.empty() || costmap.type() != CV_64FC1)
        throw std::runtime_error("planAStar: costmap must be CV_64FC1.");

    const int w = costmap.cols;
    const int h = costmap.rows;

    if (start.x < 0 || start.y < 0 || start.x >= w || start.y >= h)
        throw std::runtime_error("planAStar: start out of range.");
    if (goal.x < 0 || goal.y < 0 || goal.x >= w || goal.y >= h)
        throw std::runtime_error("planAStar: goal out of range.");

    if (isObstacle_(costmap, start.x, start.y))
        throw std::runtime_error("planAStar: start is obstacle.");
    if (isObstacle_(costmap, goal.x, goal.y))
        throw std::runtime_error("planAStar: goal is obstacle.");
    // 八方向增量
    const int dx[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
    const int dy[8] = { 0, 1, 1, 1, 0, -1, -1, -1 };
    const double step_cost[8] = {
        1.0, std::sqrt(2.0), 1.0, std::sqrt(2.0),
        1.0, std::sqrt(2.0), 1.0, std::sqrt(2.0)
    };

    cv::Mat visited(h, w, CV_8U, cv::Scalar(0));
    cv::Mat dist(h, w, CV_64FC1, cv::Scalar(1e10));
    cv::Mat parent_x(h, w, CV_32S, cv::Scalar(-1));
    cv::Mat parent_y(h, w, CV_32S, cv::Scalar(-1));

    using Node = std::tuple<double, int, int>;
    std::priority_queue<Node, std::vector<Node>, std::greater<>> open;

    open.emplace(0.0, start.x, start.y);
    dist.at<double>(start.y, start.x) = 0.0;

    while (!open.empty()) {
        auto [f, cx, cy] = open.top();
        open.pop();

        if (visited.at<uchar>(cy, cx)) continue;
        visited.at<uchar>(cy, cx) = 1;

        if (cx == goal.x && cy == goal.y) break;

        for (int dir = 0; dir < 8; ++dir) {
            int nx = cx + dx[dir];
            int ny = cy + dy[dir];

            if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
            /***** 安全区筛选 *****/
            if (!isSafe_(costmap, nx, ny, 2)) continue;
            if (visited.at<uchar>(ny, nx)) continue;
            // 使用 costmap_add_ 计算代价
            double g = dist.at<double>(cy, cx)+ step_cost[dir]+ 1000.0 * costmap.at<double>(ny, nx);

            if (g < dist.at<double>(ny, nx)) {
                dist.at<double>(ny, nx) = g;
                parent_x.at<int>(ny, nx) = cx;
                parent_y.at<int>(ny, nx) = cy;
                // 欧氏启发
                double h_cost = std::hypot(nx - goal.x, ny - goal.y);
                open.emplace(g + h_cost, nx, ny);
            }
        }
    }
    /* 回溯路径 */
    std::vector<cv::Point> path;
    int cx = goal.x;
    int cy = goal.y;

    if (parent_x.at<int>(cy, cx) == -1)
        return path;

    while (cx != start.x || cy != start.y) {
        path.emplace_back(cx, cy);
        int px = parent_x.at<int>(cy, cx);
        int py = parent_y.at<int>(cy, cx);
        cx = px;
        cy = py;
    }

    path.emplace_back(start);
    std::reverse(path.begin(), path.end());
    return path;
}