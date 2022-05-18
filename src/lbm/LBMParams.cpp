#include "LBMParams.h"
#include "stdio.h"

// ======================================================
// ======================================================
void LBMParams::setup(const ConfigMap &configMap) {

  // initialize run parameters
  maxIter = configMap.getInteger("run", "maxIter", 20000);
  outStep = configMap.getInteger("run", "outStep", 1000);
  outImage = configMap.getBool("run", "outImage", true);

  // geometry
  nx = configMap.getInteger("geometry", "nx", 1 << 9);
  ny = configMap.getInteger("geometry", "ny", 1 << 7);

  lx = static_cast<double>(nx) - 1;
  ly = static_cast<double>(ny) - 1;

  // cylinder
  cx = configMap.getFloat("cylinder", "cx", 1.0 * nx / 4);
  cy = configMap.getFloat("cylinder", "cy", 1.0 * ny / 2);
  r = configMap.getFloat("cylinder", "r", 1.0 * ny / 9);

  // fluids parameters

  // initial velocity
  uLB = configMap.getFloat("fluid", "uLB", 0.04);

  // Reynolds number.
  Re = configMap.getFloat("fluid", "Re", 150.0);

  // Viscoscity in lattice units.
  nuLB = uLB * r / Re;

  // Relaxation parameter.
  omega = 1.0 / (3 * nuLB + 0.5);

} // LBMParams::setup

// ======================================================
void LBMParams::setup(int maxIter, int outStep, int nx, int ny, double uLB,
                      double Re) {

  // initialize run parameters
  this->maxIter = maxIter;
  this->outStep = outStep;

  // geometry
  this->nx = nx;
  this->ny = ny;

  lx = static_cast<double>(nx) - 1;
  ly = static_cast<double>(ny) - 1;

  // cylinder
  cx = 1.0 * nx / 4;
  cy = 1.0 * ny / 2;
  r = 1.0 * ny / 9;

  // fluids parameters

  // initial velocity
  this->uLB = uLB;

  // Reynolds number.
  this->Re = Re;

  // Viscoscity in lattice units.
  nuLB = uLB * r / Re;

  // Relaxation parameter.
  omega = 1.0 / (3 * nuLB + 0.5);

} // LBMParams::setup

// ======================================================
// ======================================================
void LBMParams::print() {

  printf("##########################\n");
  printf("Simulation run parameters:\n");
  printf("##########################\n");
  printf("maxIter            : %d\n", maxIter);
  printf("outputStep         : %d\n", outStep);
  printf("output format      : %s\n", outImage ? "png image" : "vtk file");
  printf("nx                 : %d\n", nx);
  printf("ny                 : %d\n", ny);
  printf("lx                 : %f\n", lx);
  printf("ly                 : %f\n", ly);
  printf("cx                 : %f\n", cx);
  printf("cy                 : %f\n", cy);
  printf("r                  : %f\n", r);
  printf("uLB                : %f\n", uLB);
  printf("Re                 : %f\n", Re);
  printf("nuLB               : %f\n", nuLB);
  printf("omega              : %f\n", omega);
  printf("\n");

} // LBMParams::print
