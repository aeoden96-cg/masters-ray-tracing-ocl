cmake_minimum_required(VERSION 3.7)

project(project VERSION 0.0.1 LANGUAGES CXX)

add_executable(project
            main.cpp
            Sphere.cpp
            Sphere.hpp
        OCL.cpp OCL.h)

target_link_libraries(project
    -lGLEW -lGL -lGLU -lglut -lpthread -lm -lOpenCL
)

find_package(OpenCL REQUIRED)
target_link_libraries(project OpenCL::OpenCL)
find_package(GLEW REQUIRED)
target_link_libraries(project GLEW::GLEW)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/opencl_kernel.cl ${CMAKE_CURRENT_BINARY_DIR}/opencl_kernel.cl COPYONLY)