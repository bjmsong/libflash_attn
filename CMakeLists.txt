project(flash-attention LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CUTLASS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cutlass)

add_subdirectory(src)
