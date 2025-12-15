cmake_minimum_required (VERSION 3.31)

project (normal_modes LANGUAGES CXX)

# Compiler requirements
set(CMAKE_CXX_STANDARD 20) 
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find JSON Libraries
find_package(nlohmann_json REQUIRED)
find_package(HDF5 REQUIRED)
find_package(HighFive REQUIRED)
find_package(Armadillo REQUIRED)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(chemfiles REQUIRED)

add_library(common INTERFACE)

target_link_libraries(common INTERFACE
    Eigen3::Eigen
    armadillo
    nlohmann_json::nlohmann_json
    HDF5::HDF5
    chemfiles
)

target_include_directories(common INTERFACE ${CMAKE_SOURCE_DIR}/include)

target_compile_definitions(common INTERFACE 
    ARMA_USE_HDF5
    H5_USE_110_API
)
add_compile_options(-Wa,--gsframe=no) # This is to suppress a useless warning on x86 machines


link_libraries(common)

# If Needed Make Modifications below this line 
add_executable(normal_modes src/normal_modes.cpp src/hw3.cpp src/hw4.cpp src/hw5.cpp src/analyzer.cpp src/writter.cpp)