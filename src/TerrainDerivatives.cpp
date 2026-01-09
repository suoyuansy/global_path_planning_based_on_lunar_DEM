#include "TerrainDerivatives.hpp"
#include <stdexcept>
#include <cmath>

static constexpr double kPi = 3.14159265358979323846;

static inline double Rad2Deg(double r) { return r * 180.0 / kPi; }

static inline double Wrap360(double a) {
    a = std::fmod(a, 360.0);
    if (a < 0.0) a += 360.0;
    return a;
}

//  输出坡向 aspect：0=北，90=东，180=南，270=西（顺时针）
//   A = 270° + arctan(fy/fx) - 90° * (fx/|fx|)
SlopeAspectResult TerrainDerivatives::ComputeSlopeAspect_3rdOrder(const cv::Mat& dem_m, double g) {
    if (dem_m.empty() || dem_m.type() != CV_64FC1 || dem_m.channels() != 1) {
        throw std::runtime_error("ComputeSlopeAspect: dem_m must be CV_64FC1 single-channel.");
    }
    if (g <= 0.0) {
        throw std::runtime_error("ComputeSlopeAspect: grid size g must be > 0.");
    }

    const int rows = dem_m.rows;
    const int cols = dem_m.cols;

    //边界区域：坡度90，坡向0（正北）
    cv::Mat slope(rows, cols, CV_64FC1, cv::Scalar(90.0));
    cv::Mat aspect(rows, cols, CV_64FC1, cv::Scalar(0.0));

    // 若尺寸不足以做 3x3，直接返回全边界填充值
    if (rows < 3 || cols < 3) {
        return { slope, aspect };
    }

    const double denom = 6.0 * g;

    for (int y = 1; y < rows - 1; ++y) {
        const double* rN = dem_m.ptr<double>(y - 1);
        const double* rC = dem_m.ptr<double>(y);
        const double* rS = dem_m.ptr<double>(y + 1);

        double* srow = slope.ptr<double>(y);
        double* arow = aspect.ptr<double>(y);

        for (int x = 1; x < cols - 1; ++x) {
            // Z 编号（上北下南）
            const double Z7 = rN[x - 1];
            const double Z8 = rN[x];
            const double Z9 = rN[x + 1];

            const double Z4 = rC[x - 1];
            const double Z6 = rC[x + 1];

            const double Z1 = rS[x - 1];
            const double Z2 = rS[x];
            const double Z3 = rS[x + 1];

            // ===== 三阶差分 =====
            const double fx =
                ((Z7 - Z1) + (Z8 - Z2) + (Z9 - Z3)) / denom; // 南北
            const double fy =
                ((Z3 - Z1) + (Z6 - Z4) + (Z9 - Z7)) / denom; // 东西

            // ===== 坡度 =====
            const double grad = std::sqrt(fx * fx + fy * fy);
            srow[x] = Rad2Deg(std::atan(grad));

            // ===== 坡向 =====

            // 平坦处坡向无意义：按你的要求置 0（正北）
            if (grad < 1e-12) {
                arow[x] = 0.0;
                continue;
            }

            // fx / |fx|，fx=0 时按 +1 处理（避免未定义）
            const double sgn_fx = (fx > 0.0) ? 1.0 : (fx < 0.0 ? -1.0 : 1.0);

            // atan(fy / fx)，fx=0 时用 ±90°
            double atan_term;
            if (std::abs(fx) < 1e-15) {
                atan_term = (fy >= 0.0) ? (kPi / 2.0) : (-kPi / 2.0);
            }
            else {
                atan_term = std::atan(fy / fx);
            }

            double A = 270.0 + Rad2Deg(atan_term) - 90.0 * sgn_fx;
            A = Wrap360(A);

            arow[x] = A;
        }
    }

    return { slope, aspect };
}
