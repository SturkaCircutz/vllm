#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
// brain floating 16
#include <iostream>


int main(){
    std::cout << "vllm stage 0"<< std::endl;
    __nv_bfloat16 x = __float2bfloat16(1.0f);
    // turning float 1.0 f to bf16 save it to x
    float y = __bfloat162float(x);
    // turning it back to normal floating number
    std::cout<<y<<std::endl;
    return 0;

}