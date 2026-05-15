#pragma once

#include <cuda_runtime.h>
#include <string>
#include <stdexcept>

inline void cudaCheck(cudaError_t error, const char * expr, const char* file, int line){
    if(error != Cudasuccess){
        // runtime error is in stdexcept
        // for example CUDA error at main.cu for cudamalloc(..)  out of memory
        //expr is the error function such as cudamalloc(....)
        throw std::runtime_error{
            std::string("CUDA error at")+ file + "for"
            + expr + ": "+cudaErrorString(error);
        };
    }
}

#define CUDA_CHECK(expr) cudaCheck((expr), #expr, __FILE__, __LINE__)

