#pragma once

#include "LBMParams.h"
#include "lbmFlowUtils.h"
#include "real_type.h"
#include <CL/sycl.hpp>

using namespace cl;

// PROFILING UTILITIES
#ifdef PROFILE
#define COMPUTE_ELAPSED_TIME(name, increment, elapsedTimeMap)                  \
  computeElapsedTime(name##_ev, elapsedTimeMap[std::string(#name)], increment);

#define COMPUTE_AVG_ELAPSED_TIME(name, elapsedTimeMap, countMap)               \
  elapsedTimeMap[name] / countMap[name]
#endif

struct LBMVariables {
public:
  // Distribution functions
  real_t *fin{nullptr};
  real_t *fout{nullptr};
  real_t *feq{nullptr};

  // Macroscopic variables
  real_t *rho{nullptr};
  real_t *ux{nullptr};
  real_t *uy{nullptr};

  // Host macroscopic variables (will be allocated if needed for vtk output)
  real_t *rhoH{nullptr};
  real_t *uxH{nullptr};
  real_t *uyH{nullptr};

  // Velocity profile / Image
  real_t *u2{nullptr};
  unsigned char *img{nullptr};
  unsigned char *imgH{nullptr};

  // obstacle
  uint8_t *obstacle{nullptr};

  void AllocateVariables(const LBMParams &params, sycl::queue queue) {

    int nx = params.nx;
    int ny = params.ny;
    int npop = params.npop;

    // memory allocations
    // distribution functions
    this->fin = sycl::malloc_device<real_t>(nx * ny * npop, queue);
    this->fout = sycl::malloc_device<real_t>(nx * ny * npop, queue);
    this->feq = sycl::malloc_device<real_t>(nx * ny * npop, queue);

    // macroscopic variables
    this->rho = sycl::malloc_device<real_t>(nx * ny, queue);
    this->ux = sycl::malloc_device<real_t>(nx * ny, queue);
    this->uy = sycl::malloc_device<real_t>(nx * ny, queue);

    // output image purposes
    this->u2 = sycl::malloc_device<real_t>(nx * ny, queue);
    this->img = sycl::malloc_device<unsigned char>(nx * ny * 4, queue);
    this->imgH = new unsigned char[nx * ny * 4];

    // obstacle
    this->obstacle = sycl::malloc_device<uint8_t>(nx * ny, queue);
  }

  void FreeVariables(sycl::queue queue, bool outImage) {
    // free memory

    // Distribution functions
    sycl::free(fin, queue);
    sycl::free(fout, queue);
    sycl::free(feq, queue);

    // Macroscopic variables

    sycl::free(rho, queue);
    sycl::free(ux, queue);
    sycl::free(uy, queue);

    // Velocity profile image
    sycl::free(u2, queue);
    sycl::free(img, queue);

    // Obstacle
    sycl::free(obstacle, queue);

    // Free allocated duplicates of rho, ux and uy (host side)
    // when vtk output used
    if (!outImage) {
      delete[] uxH;
      delete[] uyH;
      delete[] rhoH;
      delete[] imgH;
    }
  }
}; // Struct LBMVariables

/**
 * class LBMSolver for D2Q9
 *
 * Adapted and translated to C++ from original python version
 * found here :
 * https://github.com/sidsriv/Simulation-and-modelling-of-natural-processes
 *
 * LBM lattice : D2Q9
 *
 * 6   3   0
 *  \  |  /
 *   \ | /
 * 7---4---1
 *   / | \
 *  /  |  \
 * 8   5   2
 *
 */
class LBMSolver {

public:
  LBMSolver(const LBMParams &params, sycl::device device);
  ~LBMSolver();

  // LBM weight for D2Q9
  const real_t tHost[9] = {1.0 / 36, 1.0 / 9,  1.0 / 36, 1.0 / 9, 4.0 / 9,
                           1.0 / 9,  1.0 / 36, 1.0 / 9,  1.0 / 36};

  // LBM lattive velocity (X and Y components) for D2Q9
  const real_t vHost[9 * 2]{1, 1, 1,  0,  1, -1, 0, 1,  0,
                            0, 0, -1, -1, 1, -1, 0, -1, -1};

  real_t *t{nullptr};
  real_t *v{nullptr};

  // LBM Params
  const LBMParams &params;

  // Variables
  LBMVariables lbm_vars;

  // SYCL Queue
  sycl::queue queue;

  // Base methods
  void initialize();
  void run();

  // Output methods
  void output_png(int iTime);
  void output_vtk(int iTime);

  // SYCL Events - Kernel-wise
  // All events needed for synchronization & profiling (if enabled)
  sycl::event init_obstacle_mask_ev, copy_t_ev, copy_v_ev,
      initialize_equilibrium_ev;
  std::vector<sycl::event> initialize_macroscopic_variables_ev;

  sycl::event border_outflow_ev, border_inflow_ev, update_fin_inflow_ev;
  sycl::event macroscopic_ev, equilibrium_ev, update_obstacle_ev;
  sycl::event compute_collision_ev, streaming_ev;

  std::vector<sycl::event> prepare_png_output_ev;

  // SYCL device info
  void printQueueInfo();

  // Profiling infos (if enabled)

  // PROFILING Data Placeholders
  // Elapsed time per routine
  std::map<std::string, double> elapsedTime{
      // Single execution routines
      {"init_obstacle_mask", 0.0},
      {"copy_t", 0.0},
      {"copy_v", 0.0},
      {"initialize_macroscopic_variables", 0.0},
      {"initialize_equilibrium", 0.0},
      // Simulation-Steps repeated routines
      {"equilibrium", 0.0},
      {"border_outflow", 0.0},
      {"macroscopic", 0.0},
      {"border_inflow", 0.0},
      {"update_fin_inflow", 0.0},
      {"compute_collision", 0.0},
      {"update_obstacle", 0.0},
      {"streaming", 0.0},
      // Simulation-Steps / outputStep repeated routine
      {"prepare_png_output", 0.0},
  };

  // Repetition call count per routine
  std::map<std::string, int> repetitionCount{
      // Single execution routines
      {"init_obstacle_mask", 1},
      {"copy_t", 1},
      {"copy_v", 1},
      {"initialize_macroscopic_variables", 1},
      {"initialize_equilibrium", 1},
      // Simulation-Steps repeated routines
      {"equilibrium", params.maxIter},
      {"border_outflow", params.maxIter},
      {"macroscopic", params.maxIter},
      {"border_inflow", params.maxIter},
      {"update_fin_inflow", params.maxIter},
      {"compute_collision", params.maxIter},
      {"update_obstacle", params.maxIter},
      {"streaming", params.maxIter},
      // Simulation-Steps / outputStep repeated routine
      {"prepare_png_output", params.maxIter / params.outStep},
  };

}; // class LBMSolver
