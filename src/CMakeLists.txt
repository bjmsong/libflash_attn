# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --expt-relaxed-constexpr --use_fast_math -t 30 \
#                       -gencode=arch=compute_75,code=\\\"sm_75,compute_75\\\" \
#                       ")
                      
set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17 --expt-relaxed-constexpr --use_fast_math")
if (${CMAKE_BUILD_TYPE} MATCHES "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_${CMAKE_CUDA_ARCHITECTURES} -g -lineinfo -O0")
else ()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${CMAKE_CUDA_ARCHITECTURES},code=sm_${CMAKE_CUDA_ARCHITECTURES} --use_fast_math -O3")
endif ()

                      
find_package (CUDA REQUIRED)
find_package(gflags REQUIRED)

set (SYS_CUDART_PATH "/usr/local/cuda")
set (SYS_CUDA_DRIVER_PATH "/usr/lib/x86_64-linux-gnu")
include_directories(../include
                    ${CUTLASS_DIR}/include
                    ${SYS_CUDART_PATH}/include
                    ${GFLAGS_INCLUDE_DIR})
link_directories (
  ${SYS_CUDART_PATH}/lib64
  ${SYS_CUDA_DRIVER_PATH}
)

file(GLOB SRCS 
  flash.cu
  flash_fwd_hdim128_fp16_sm80.cu
  flash_fwd_hdim160_fp16_sm80.cu
  flash_fwd_hdim192_fp16_sm80.cu
  flash_fwd_hdim224_fp16_sm80.cu
  flash_fwd_hdim256_fp16_sm80.cu
  flash_fwd_hdim32_fp16_sm80.cu
  flash_fwd_hdim64_fp16_sm80.cu
  flash_fwd_hdim96_fp16_sm80.cu
)
set(CMAKE_CUDA_ARCHITECTURES "native")

add_executable(main main.cpp ${SRCS})
target_link_libraries(main ${GFLAGS_LIBRARIES})