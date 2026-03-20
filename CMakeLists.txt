# 告诉 CMake 最低可用版本；VS2019/2022 自带 3.20+，写 3.16 足够
cmake_minimum_required(VERSION 3.16)

#  定义“工程名”+指定语言；这里语言只用 C++（CXX）
project(path_planning_based_on_lunar_DEM LANGUAGES CXX)

#  把 C++ 标准一次性升到 17，省得以后每个 target 都写
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)   # 强制必须支持 17，否则报错


#  告诉 CMake 去 third_party/opencv/build 里找 OpenCVConfig.cmake
set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/opencv/build")
find_package(OpenCV REQUIRED)

#  先把“可执行文件”声明出来，名字一般跟工程名一致
#  目前只放了一个空 main.cpp，保证能编译通过
add_executable(${PROJECT_NAME}        # ${} 是取变量，这里 = 上面 project() 的名字
    src/main.cpp
   "src/Dem.cpp" "src/TerrainSlopeAspect.cpp" "src/TerrainRoughness.cpp" "src/TerrainStepEdge.cpp" "src/TerrainObstacleExpand.cpp" "src/TerrainCostmapFusion.cpp" "src/PathPlanningInteractive.cpp" "src/PathPlanning_Local_API.cpp" "src/PathPlanner.cpp" "src/PathPlanning_Global_API.cpp")

# 把头文件搜索路径告诉编译器，这样写 #include "xxx.h" 时不用写相对路径
target_include_directories(${PROJECT_NAME}
    PRIVATE                           # 仅这个靶标（即可执行文件）可见
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}      # OpenCV 头文件路径
)

#收集 OpenCV 所有 dll（路径随 OpenCV_DIR 自动定位）
file(GLOB OPENCV_DLLS
     "${OpenCV_DIR}/x64/vc16/bin/*.dll")

#构建完成后复制到 exe 同级目录
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${OPENCV_DLLS}
            $<TARGET_FILE_DIR:${PROJECT_NAME}>
    COMMENT "Copying OpenCV dlls to output directory ..."
)
#  把 OpenCV 的库文件链接到 exe
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        ${OpenCV_LIBS}
)

# 把 data 整个文件夹复制到生成目录，调试时就能直接用相对路径
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_SOURCE_DIR}/data"
    "$<TARGET_FILE_DIR:${PROJECT_NAME}>/data"
    COMMENT "Copying data folder to output directory ..."
)