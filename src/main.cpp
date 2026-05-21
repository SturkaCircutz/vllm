#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
// brain floating 16
#include <iostream>


int main(){
    vllm::ModelConfig cfg = vllm::tinyConfig();

    if(!vllm::validateModelConfig(cfg)){
        std::cerr<<"invalid";
        return 1;
    }

    std::cout<<"hidden dim: "<<cfg.hidden_dim<<endl;
    std::cout<<"num_heads: "<<cfg.num_heads<<endl;
    std::cout<<"head_dim: "<<cfg.head_dim<<endl;
    vllm::DeviceBuffer<__nv_bfloat16> buffer(1024);
    std::cout<<"Device buffer size: "<<buffer.size()<<endl;
    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;

}