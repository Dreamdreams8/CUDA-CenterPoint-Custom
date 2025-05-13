cmake_minimum_required(VERSION 3.14)
project(centerpoint)
set(arch ${CMAKE_HOST_SYSTEM_PROCESSOR})

if(${arch}     STREQUAL  "aarch64")
  set(TENSORRT_INCLUDE /usr/include/aarch64-linux-gnu/)
  set(TENSORRT_LIB     /usr/lib/aarch64-linux-gnu/)  
elseif(${arch} STREQUAL "x86_64")
  set(TENSORRT_INCLUDE /mnt/why/soft/TensorRT-8.5.3.1/include)
  set(TENSORRT_LIB     /usr/local/cuda-11.6/lib)  
endif()


find_package(Protobuf REQUIRED)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-std=c++11 -Wextra -Wall -Wno-deprecated-declarations -O3")


find_package(CUDA REQUIRED)
if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  message("Using Debug Mode")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -g -G --ptxas-options=-v)
endif()

include_directories(
  ${CUDA_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE}
  src/libspconv/include
  src
  ${PROTOBUF_INCLUDE_DIRS}
)

link_directories(
  ${TENSORRT_LIB}
  src/libspconv/lib/${arch}
)


file(GLOB_RECURSE SOURCE_FILES 
  src/*.cu 
  src/*.cpp
  src/onnx/*.cpp
)

# 创建共享库而不是可执行文件
cuda_add_executable(${PROJECT_NAME} main.cpp ${SOURCE_FILES})

target_link_LIBraries(${PROJECT_NAME}
    libnvinfer.so
    libspconv.so
    ${Protobuf_LIBRARIES}
)


