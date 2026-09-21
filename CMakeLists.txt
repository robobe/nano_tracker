cmake_minimum_required(VERSION 3.28)

project(nanotracker_cpp LANGUAGES CXX)

option(NANOTRACKER_BUILD_ORT "Build the ONNX Runtime example" ON)
option(NANOTRACKER_BUILD_NCNN "Build the NCNN CPU example" OFF)
option(NANOTRACKER_BUILD_RKNN "Build the RKNN example" OFF)

add_subdirectory(cpp)
