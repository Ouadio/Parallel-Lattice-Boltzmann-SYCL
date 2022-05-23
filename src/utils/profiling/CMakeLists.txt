add_library(profiling "")
target_sources(profiling PRIVATE
  sycl_profiler_utils.cpp sycl_profiler_utils.h)

target_include_directories(profiling PRIVATE "${SYCL_INCLUDE_DIRS}")

if(USE_COMPUTECPP)
  set_target_properties(profiling PROPERTIES INTERFACE_LINK_LIBRARIES ComputeCpp::ComputeCpp )
else()
  set_target_properties(profiling PROPERTIES INTERFACE_LINK_LIBRARIES DPCPP::DPCPP )
  target_link_libraries(profiling PRIVATE DPCPP::DPCPP)
endif()

add_sycl_to_target(TARGET profiling SOURCES)

add_library(lbm::profiling ALIAS profiling)
