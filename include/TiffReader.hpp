#pragma once
#include "Dem.hpp"
#include <string>

class TiffReader {
public:
    // 读取单通道 32-bit TIFF 到 dem.raw()
    static void ReadSingleChannel32Bit(const std::string& tiff_path, Dem& dem);
};
