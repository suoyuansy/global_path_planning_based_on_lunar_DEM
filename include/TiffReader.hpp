#pragma once
#include "Dem.hpp"
#include <string>

class TiffReader {
public:
    static Dem ReadSingleChannel32Bit(const std::string& tiff_path);
};
