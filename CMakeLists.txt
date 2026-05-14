cmake_minimum_required(VERSION 3.24) 
# make sure that cmake is at least 3.24

project(vllm LANGUAGES CXX CUDA)


#set cpp standard as cpp 17
set(CMAKE_CXX_STANDARD 17)

#must be cpp 17
set(CMAKE_CXX_STANDARD_REQUIRED ON)



# if didnt specify build's type, use release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)   # NVIDIA RTX 40 arch
endif()

find_package(CUDAToolkit REQUIRED)


#create an executable named vllm
add_executable(vllm
    src/main.cpp    
)

#tell vllm the header is in include
target_include_directories(vllm PRIVATE include)

#private, this is only for the vllm


#connect rt and cublas to vllm
target_link_libraries(vllm PRIVATE 
CUDA::cudart
CUDA::cublas)

#private is only used for the executable, vllm not for other targets



