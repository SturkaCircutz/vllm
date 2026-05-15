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
                if(this != & other){
                    reset();
                    p
                }
            }
    }
}