# Project Entry CMakeLists.txt 

cmake_minimum_required(VERSION 3.10)
project(LBMcpp LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

# custom cmake macros location
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake/sycl")

# Prevent from build in source tree
include(preventBuildInSource)

# SYCL Related

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-unnamed-lambda -std=c++14 -D__SYCL_DISABLE_NAMESPACE_INLINE__=ON -Xclang -cl-mad-enable") # [-O3] for further optimizations

# BUILD OPTIONS

# SYCL Device (GPU | CPU)
option(USE_GPU "Use Available GPU as Computation Offload Device" OFF)

if(USE_GPU)
    set (DEVICE_MSSG "++ Build to use GPU as SYCL Device")
    add_definitions(-DSYCL_GPU)
    set(DPCPP_SYCL_TARGET spir64-unknown-unknown-sycldevice)         #GPU 
else()
    set( DEVICE_MSSG "++ Build to use CPU as SYCL Device")
    set(DPCPP_SYCL_TARGET spir64_x86_64-unknown-unknown-sycldevice)  #CPU
endif(USE_GPU)

# SYCL PROFILING (ENABLED | DISABLED)
option(ENABLE_PROFILING "Enable SYCL profiling of the application" OFF)

if(ENABLE_PROFILING)
  set(PROFILING_MSSG "++ Build with profiling enabled")
  add_definitions(-DPROFILE)
else()
  set(PROFILING_MSSG "")
endif(ENABLE_PROFILING)

# Type to use (single precision | double precision)
option (USE_DOUBLE "build with double precision" OFF)

if (USE_DOUBLE)
  add_compile_options(-DUSE_DOUBLE)
  set(TYPE_MSSG "++ Build with DOUBLE precision as base type")
else()
  set(TYPE_MSSG "++ Build with SINGLE precision as base type")
endif(USE_DOUBLE)

find_package(DPCPP REQUIRED)

get_target_property(SYCL_INCLUDE_DIRS DPCPP::DPCPP INTERFACE_INCLUDE_DIRECTORIES)

# build some executable
add_subdirectory(src)

#-------------------------------------------------------------------#
# Build Configuration summary
#-------------------------------------------------------------------#
message("  ===================================================")
message("  ${PROJECT_NAME} build configuration:")
message("  ===================================================")
message("  C++ Compiler : ${CMAKE_CXX_COMPILER_ID} "
        "${CMAKE_CXX_COMPILER_VERSION} ")
message("                 ${CMAKE_CXX_COMPILER}")
message("  CXX Compilation Flags  :  ${CMAKE_CXX_FLAGS}")
message("  ---------------------------------------------------")
message("  ${DEVICE_MSSG}")
message("  ${PROFILING_MSSG}")
message("  ${TYPE_MSSG}")
message("  ---------------------------------------------------")
message("")

