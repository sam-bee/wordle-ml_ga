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

include(cmake/word_data.cmake)
add_library(wordle_fitness STATIC src/fitness/vocabulary.cpp src/fitness/game.cu src/fitness/evaluator.cu)
target_include_directories(wordle_fitness PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(wordle_fitness PUBLIC wordle_policy genotype_slab)
target_compile_options(wordle_fitness PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra>
)

add_executable(policy_batch_test tests/cuda/policy_batch_test.cu)
target_link_libraries(policy_batch_test PRIVATE wordle_policy)
target_include_directories(policy_batch_test PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_compile_definitions(policy_batch_test PRIVATE POLICY_FIXTURE_DIR="${POLICY_FIXTURE_DIR}")
target_compile_options(policy_batch_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)
add_test(NAME policy_batch_test COMMAND policy_batch_test)
set_tests_properties(policy_batch_test PROPERTIES LABELS gpu TIMEOUT 30)

add_executable(fitness_test tests/cuda/fitness_test.cu)
target_link_libraries(fitness_test PRIVATE wordle_fitness)
target_include_directories(fitness_test PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_compile_definitions(fitness_test PRIVATE
    POLICY_FIXTURE_DIR="${POLICY_FIXTURE_DIR}"
    WORD_DATA_DIR="${WORD_DATA_DIR}"
)
target_compile_options(fitness_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)
add_test(NAME fitness_test COMMAND fitness_test)
set_tests_properties(fitness_test PROPERTIES LABELS gpu TIMEOUT 60)

add_executable(fitness_runtime_test tests/cuda/fitness_runtime_test.cu)
target_link_libraries(fitness_runtime_test PRIVATE wordle_fitness)
target_compile_definitions(fitness_runtime_test PRIVATE WORD_DATA_DIR="${WORD_DATA_DIR}")
target_compile_options(fitness_runtime_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)
add_test(NAME fitness_runtime_test COMMAND fitness_runtime_test)
set_tests_properties(fitness_runtime_test PROPERTIES LABELS gpu TIMEOUT 120)

add_executable(fitness_game_test tests/cuda/fitness_game_test.cu)
target_link_libraries(fitness_game_test PRIVATE wordle_fitness)
target_include_directories(fitness_game_test PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_compile_definitions(fitness_game_test PRIVATE
    POLICY_FIXTURE_DIR="${POLICY_FIXTURE_DIR}"
    WORD_DATA_DIR="${WORD_DATA_DIR}"
)
target_compile_options(fitness_game_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)
add_test(NAME fitness_game_test COMMAND fitness_game_test)
set_tests_properties(fitness_game_test PROPERTIES LABELS gpu TIMEOUT 60)
