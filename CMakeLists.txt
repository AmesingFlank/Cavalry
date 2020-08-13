cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(Cavalry LANGUAGES CXX CUDA)


set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math")

set(CMAKE_CXX_STANDARD 17)



add_executable(Cavalry
 main.cpp
)

IF(APPLE)
set_property(TARGET Cavalry
        PROPERTY
        BUILD_RPATH /usr/local/cuda/lib)
ENDIF(APPLE)

set_property(TARGET Cavalry PROPERTY CUDA_STANDARD 11)
