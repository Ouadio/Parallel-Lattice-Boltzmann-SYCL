add_library(saveVTK "")
target_sources(saveVTK PRIVATE
  saveVTK.cpp)

target_include_directories(saveVTK
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  )

add_library(lbm::vtk ALIAS saveVTK)

