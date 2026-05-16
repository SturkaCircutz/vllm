#pragma once
#include "cuda_check.hpp"
#include <cstddef>
#include <utility>


// raii 
namespace vllm{

    template<typename T>
    class DeviceBuffer{
        public:
            DeviceBuffer() = default;
            // dont allow implicit conventions

            // DeviceBuffer<float> buf(1024) for example
            explicit DeviceBuffer(size_t count){
                allocate(count);
            }

            // when the function is finished, st cudafree(ptr), ptr = nullptr, count = 0;
            ~DeviceBuffer{
                reset();
            }

            DeviceBuffer(const DeviceBuffer&) = delete;
            // dont allow aliasing, shallow copy only makes the aliasing, 2 pointers use the same gpu mem
            // which would cause double free
            // DeviceBuffer<float> a(1024);
            // DeviceBuffer<float> b(a); 
            DeviceBuffer& operator=(const DeviceBuffer&) = delete;
            //
            // DeviceBuffer<float> a(1024);
            // DeviceBuffer<float> b(1024); 
            // a = b; will call a.operator=b;
            // return *this which is b.
            DeviceBuffer(DeviceBuffer&& other) noexcept
                : ptr_(std::exchange(other.ptr_, nullptr)),

                // ptr_ = other.ptr_
                // otherptr= nullptr
                size_(std::exchange(other.size_, 0)){}
                //size_=other.size
                // other.size = 0

            DeviceBuffer& operator=(DeviceBuffer && other) noexcept
            {

//DeviceBuffer<float> a(1024);
//DeviceBuffer<float> b(2048);
// b = std::move(a);
// a.ptr GPU mem 0xaaa
// a.size = 1024
// b.ptr gpu mem 0xbbb
// b,size 2048

//exchange b.ptr = a.ptr
// b size = 1024, a size = 0
                if(this != & other){
                    reset();
                    ptr_ = std::exchange(other.ptr_, nullptr);
                    size_ = std::exchange(other.size, 0);
                }
                return *this;
            }


            // allocate the mem
            void allocate(size_t count){
                reset();

                if(count==0) return 0;

                CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
                size_ = count;
            }

            // reset the mem
            void reset(){
                if(ptr_ != nullptr){
                    CUDA_CHECK(cudaFree(ptr_));
                    ptr_ = nullptr;
                    size_ = 0;
                }
            }

            T*data() return ptr_; // return gpu buffer row pointer
            const T*data() const return ptr_;
            size_t size() const {
                return size_;
            }

            private:
            T*ptr_ = nullptr;
            size_t size_ = 0;


    };
}