cmake_minimum_required(VERSION 3.15...3.27)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)

# Workaround for CMake not finding nanobind in Nix build
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
# End workaround

find_package(CUDAToolkit REQUIRED)
find_package(GSL REQUIRED)
find_package(nanobind CONFIG REQUIRED)

enable_language(CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20 -Wall -Werror")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++20 -Wall -Werror all-warnings")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# # Debug
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -lineinfo")
# # End debug

nanobind_add_module(cpu
  src/ising_mcmc/hypercube.cpp
  src/ising_mcmc/cpu.cpp
  src/ising_mcmc/cpu/fm.cpp
)

nanobind_add_module(cuda
  src/ising_mcmc/hypercube.cpp
  src/ising_mcmc/cuda.cu
  src/ising_mcmc/cuda/fm.cu
  src/ising_mcmc/cuda/fm_2d.cu
)

# add project root to the include path
target_include_directories(cpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/ising_mcmc)
target_include_directories(cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/ising_mcmc)

target_link_libraries(cpu PRIVATE GSL::gsl GSL::gslcblas)
target_link_libraries(cuda PRIVATE CUDA::curand)

install(TARGETS cpu cuda LIBRARY DESTINATION ising_mcmc)
