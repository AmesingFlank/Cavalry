cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(Cavalry LANGUAGES CXX CUDA C)


set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math -std=c++17 -Xcompiler \"/std:c++17\"")
set(CMAKE_CXX_STANDARD 17)


include_directories(Dependencies/include)

file(GLOB ALL_CODE
    "main.cu"
    "BSDFs/*.h"
    "BSDFs/*.cpp"
    "BSDFs/*.cu"
    "Cameras/*.h"
    "Cameras/*.cpp"
    "Cameras/*.cu"
    "Core/*.h"
    "Core/*.cpp"
    "Core/*.cu"
    "Dependencies/*.h"
    "Dependencies/*.inc"
    "Films/*.h"
    "Films/*.cpp"
    "Films/*.cu"
    "Integrators/*.h"
    "Integrators/*.cpp"
    "Integrators/*.cu"
    "Samplers/*.h"
    "Samplers/*.cpp"
    "Samplers/*.cu"
    "Lights/*.h"
    "Lights/*.cpp"
    "Lights/*.cu"
    "Shapes/*.h"
    "Shapes/*.cpp"
    "Shapes/*.cu"
    "Utils/*.h"
    "Utils/*.cpp"
    "Utils/*.cu"
)

add_executable(Cavalry ${ALL_CODE})




set_property(TARGET Cavalry PROPERTY CUDA_STANDARD 17)

IF(APPLE)
set_property(TARGET Cavalry
        PROPERTY
        BUILD_RPATH /usr/local/cuda/lib)
ENDIF(APPLE)