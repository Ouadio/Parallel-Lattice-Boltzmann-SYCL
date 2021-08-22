# General Src CMakeLists.txt

add_subdirectory(utils)
add_subdirectory(lbm)

add_executable(runLBMSimulation "")

target_sources(runLBMSimulation PRIVATE
  main.cpp)

target_include_directories(runLBMSimulation
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src)

set_target_properties(runLBMSimulation PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${SYCL_INCLUDE_DIRS}")
set_target_properties(runLBMSimulation PROPERTIES INTERFACE_LINK_LIBRARIES DPCPP::DPCPP )

target_link_libraries(runLBMSimulation PRIVATE DPCPP::DPCPP)

target_link_libraries(runLBMSimulation
  PUBLIC
  lbm::core
  lbm::config
  lbm::lodepng
  lbm::vtk)

add_sycl_to_target(TARGET runLBMSimulation SOURCES )

configure_file(flowAroundCylinder.ini flowAroundCylinder.ini COPYONLY)
