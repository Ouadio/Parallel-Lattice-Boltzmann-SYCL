add_library(profiling "")
target_sources(profiling PRIVATE
  sycl_profiler_utils.cpp)

add_library(lbm::profiling ALIAS profiling)
