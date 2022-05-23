#pragma once

#include "utils/config/ConfigMap.h"

/**
 * LBM Parameters (declaration)
 */
template <typename T> struct LBMParams {

  //! dimension : 2 or 3
  static const int dim = 2;

  //! number of populations (of distribution functions)
  static const int npop = 9;

  //! run parameters
  int maxIter;
  int outStep;
  bool outImage;

  //! geometry : number of nodes along X axis
  int nx;

  //! geometry : number of nodes along Y axis
  int ny;

  //! physical domain sizes (in lattice units) along X axis
  T lx;

  //! physical domain sizes (in lattice units) along Y axis
  T ly;

  // cylinder obstacle (center coordinates, radius)

  //! x coordinates of cylinder center
  T cx;

  //! y coordinates of cylinder center
  T cy;

  //! cylinder radius
  T r;

  //! initial velocity
  T uLB;

  /*
   * fluid parameters
   */
  //! Reynolds number
  T Re;

  //! viscosity in lattice units
  T nuLB;

  //! relaxation parameter
  T omega;

  //! setup / initialization
  void setup(const ConfigMap &configMap);
  void setup(int maxIter, int outputStep, int nx, int ny, T uLB, T Re);

  //! print parameters on screen
  void print();

}; // struct LBMParams
