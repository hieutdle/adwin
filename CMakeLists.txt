cmake_minimum_required(VERSION 3.18)

project(adwin)

add_subdirectory(third_party)

set(ADWIN_SRC_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/src/include)
include_directories(${ADWIN_SRC_INCLUDE_DIR})


pybind11_add_module(adwin src/adwin.cc)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

