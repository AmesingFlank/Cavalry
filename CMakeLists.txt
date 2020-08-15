cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(Cavalry LANGUAGES CXX CUDA)


set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math")

set(CMAKE_CXX_STANDARD 17)

include_directories(Dependencies/include)

add_executable(Cavalry
 main.cpp
 Core/Camera.h
 Core/Film.h
 Core/Integrator.h
 Core/Renderer.cpp
 Core/Renderer.h
 Core/RenderResult.cpp
 Core/RenderResult.h
 Core/Scene.h
)

IF(APPLE)
set_property(TARGET Cavalry
        PROPERTY
        BUILD_RPATH /usr/local/cuda/lib)
ENDIF(APPLE)

set_property(TARGET Cavalry PROPERTY CUDA_STANDARD 11)
