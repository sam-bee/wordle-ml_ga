cmake_minimum_required(VERSION 3.22)

# Both development GPUs target compute capability 12.0.
set(CMAKE_CUDA_ARCHITECTURES 120)

project(wordle_ga LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

enable_testing()

add_executable(gpu_smoke_test tests/cuda/gpu_smoke_test.cu)
target_compile_options(gpu_smoke_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
)

add_test(NAME gpu_smoke_test COMMAND gpu_smoke_test)
set_tests_properties(gpu_smoke_test PROPERTIES LABELS gpu TIMEOUT 30)
