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

find_package(CUDAToolkit REQUIRED)

add_library(wordle_policy STATIC src/model/policy.cu)
target_include_directories(wordle_policy PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(wordle_policy PUBLIC CUDA::cudart)
target_compile_options(wordle_policy PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)

add_executable(gpu_smoke_test tests/cuda/gpu_smoke_test.cu)
target_compile_options(gpu_smoke_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
)

add_test(NAME gpu_smoke_test COMMAND gpu_smoke_test)
set_tests_properties(gpu_smoke_test PROPERTIES LABELS gpu TIMEOUT 30)

include(tests/policy_fixtures.cmake)
add_executable(policy_test tests/cuda/policy_test.cu)
target_link_libraries(policy_test PRIVATE wordle_policy)
target_include_directories(policy_test PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_compile_definitions(policy_test PRIVATE POLICY_FIXTURE_DIR="${POLICY_FIXTURE_DIR}")
target_compile_options(policy_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)
add_test(NAME policy_test COMMAND policy_test)
set_tests_properties(policy_test PROPERTIES LABELS gpu TIMEOUT 30)

add_library(genotype_slab STATIC src/genotype_slab/slab.cu)
target_include_directories(genotype_slab PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(genotype_slab PUBLIC CUDA::cudart)
target_compile_options(genotype_slab PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)

add_executable(genotype_slab_test tests/cuda/genotype_slab_test.cu)
target_link_libraries(genotype_slab_test PRIVATE genotype_slab)
target_compile_options(genotype_slab_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)
add_test(NAME genotype_slab_test COMMAND genotype_slab_test)
set_tests_properties(genotype_slab_test PROPERTIES LABELS gpu TIMEOUT 30)
