# LBM Core Implementation CMakeLists.txt

add_library(lbmCore "")
target_sources(lbmCore PRIVATE
  LBMParams.cpp
  lbmFlowUtils.cpp
  LBMSolver.cpp)

target_include_directories(lbmCore
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/src/utils)

set_target_properties(lbmCore PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${SYCL_INCLUDE_DIRS}")
set_target_properties(lbmCore PROPERTIES INTERFACE_LINK_LIBRARIES DPCPP::DPCPP )

target_link_libraries(lbmCore PRIVATE DPCPP::DPCPP)

add_sycl_to_target(TARGET lbmCore SOURCES )

add_library(lbm::core ALIAS lbmCore)
