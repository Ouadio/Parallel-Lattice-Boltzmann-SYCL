#pragma once

#include "LBMParams.h"
#include "profiling/sycl_profiler_utils.h"
#include <CL/sycl.hpp>

using namespace cl;

// PROFILING UTILITIES
#ifdef PROFILE
#define COMPUTE_ELAPSED_TIME(name, increment, elapsedTimeMap)                  \
  computeElapsedTime(name##_ev, elapsedTimeMap[std::string(#name)], increment);

#define COMPUTE_AVG_ELAPSED_TIME(name, elapsedTimeMap, countMap)               \
  elapsedTimeMap[name] / countMap[name]
#endif

template <typename T> struct LBMVariables {
public:
  // Distribution functions

  // Host macroscopic variables (will be allocated if needed for vtk output)

  // LBM weight for D2Q9
  const T tHost[9] = {1.0 / 36, 1.0 / 9,  1.0 / 36, 1.0 / 9, 4.0 / 9,
                      1.0 / 9,  1.0 / 36, 1.0 / 9,  1.0 / 36};

  // LBM lattive velocity (X and Y components) for D2Q9
  const T vHost[9 * 2]{1, 1, 1,  0,  1, -1, 0, 1,  0,
                       0, 0, -1, -1, 1, -1, 0, -1, -1};

  T *rhoH{nullptr};
  T *uxH{nullptr};
  T *uyH{nullptr};

  sycl::buffer<T, 1> fout{nullptr, sycl::range<1>{0}};

  sycl::buffer<T, 1> fin{nullptr, sycl::range<1>{0}};
  sycl::buffer<T, 1> feq{nullptr, sycl::range<1>{0}};

  // Constant buffers

  sycl::buffer<T, 1> t{nullptr, sycl::range<1>{0}};
  sycl::buffer<T, 1> v{nullptr, sycl::range<1>{0}};

  // Variables Buffers
  sycl::buffer<uint8_t, 1> obstacleBuff{nullptr, sycl::range<1>{0}};

  sycl::buffer<T, 1> rho{nullptr, sycl::range<1>{0}};
  sycl::buffer<T, 1> ux{nullptr, sycl::range<1>{0}};
  sycl::buffer<T, 1> uy{nullptr, sycl::range<1>{0}};

  sycl::buffer<T, 1> u2{nullptr, sycl::range<1>{0}};
  sycl::buffer<unsigned char, 1> img{nullptr, sycl::range<1>{0}};
  sycl::buffer<T, 1> max{nullptr, sycl::range<1>{0}};
  sycl::buffer<T, 1> min{nullptr, sycl::range<1>{0}};

  unsigned char *imgH{nullptr};

  bool outImage;
  // obstacle
  // sycl::buffer<uint8_t, 1> obstacleBuff{nullptr, sycl::range<1>{0}};

  LBMVariables(const LBMParams<T> &params) {

    const size_t nx = params.nx;
    const size_t ny = params.ny;
    const size_t npop = params.npop;

    outImage = params.outImage;

    fout = sycl::buffer<T, 1>(sycl::range<1>{nx * ny * npop});
    fin = sycl::buffer<T, 1>(sycl::range<1>{nx * ny * npop});
    feq = sycl::buffer<T, 1>(sycl::range<1>{nx * ny * npop});

    // Constant buffers

    t = sycl::buffer<T, 1>{tHost, sycl::range<1>{npop}};
    v = sycl::buffer<T, 1>{vHost, sycl::range<1>{2 * npop}};

    // Variables Buffers
    obstacleBuff = sycl::buffer<uint8_t, 1>(sycl::range<1>{nx * ny});

    rho = sycl::buffer<T, 1>(sycl::range<1>{nx * ny});
    ux = sycl::buffer<T, 1>(sycl::range<1>{nx * ny});
    uy = sycl::buffer<T, 1>(sycl::range<1>{nx * ny});

    u2 = sycl::buffer<T, 1>(sycl::range<1>{nx * ny});
    img = sycl::buffer<unsigned char, 1>(sycl::range<1>{nx * ny * 4});

    max = sycl::buffer<T, 1>(sycl::range<1>{1});
    min = sycl::buffer<T, 1>(sycl::range<1>{1});

    imgH = new unsigned char[nx * ny * 4];

    if (!params.outImage) {
      rhoH = new T[nx * ny];
      uxH = new T[nx * ny];
      uyH = new T[nx * ny];
    }
  }

  ~LBMVariables() {
    delete[] imgH;

    if (!outImage) {
      // Free allocated duplicates of rho, ux and uy (host side)
      // when vtk output used
      delete[] uxH;
      delete[] uyH;
      delete[] rhoH;
    }
  }

}; // Struct LBMVariables

/**
 * class LBMSolver for D2Q9
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
template<typename T> class LBMSolver {

public:
  LBMSolver(const LBMParams<T> &params, sycl::device device);

  // LBM Params
  const LBMParams<T> &params;

  // Variables
  LBMVariables<T> lbm_vars;

  // SYCL Queue
  sycl::queue queue;

  // Base methods
  void run();

  // Output methods
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

// Kernels

template <typename T, typename I> struct ObstacleKernel {
  using syclDiscardWriteAccessor =
      sycl::accessor<I, 1, sycl::access::mode::discard_write>;

public:
  ObstacleKernel(const LBMParams<T> &params, syclDiscardWriteAccessor obstacleAcc)
      : obsAcc_{obstacleAcc} {
    nx_ = params.nx;
    cx_ = params.cx;
    cy_ = params.cy;
    r2_ = params.r * params.r;
  }

  void operator()(sycl::id<2> id) const {
    size_t i = id[1]; // along nx
    size_t j = id[0]; // along ny

    size_t index = i + nx_ * j;

    T x = 1.0 * i;
    T y = 1.0 * j;

    obsAcc_[index] =
        (x - cx_) * (x - cx_) + (y - cy_) * (y - cy_) < r2_ ? 1 : 0;
  }

private:
  syclDiscardWriteAccessor obsAcc_;
  T nx_;
  T cx_;
  T cy_;
  T r2_;
};

template <typename T> struct InitEquilibrium {
  using syclDiscardWriteAccessor =
      sycl::accessor<T, 1, sycl::access::mode::discard_write>;

  using syclReadAccessor = sycl::accessor<T, 1, sycl::access::mode::read>;
  using syclConstReadAccessor =
      sycl::accessor<T, 1, sycl::access::mode::read,
                     sycl::access::target::constant_buffer>;

public:
  InitEquilibrium(const LBMParams<T> &params, syclConstReadAccessor vAcc,
                  syclConstReadAccessor tAcc, syclReadAccessor uxAcc,
                  syclReadAccessor rhoAcc, syclDiscardWriteAccessor finAcc)
      : vAcc_{vAcc}, tAcc_{tAcc}, uxAcc_{uxAcc}, rhoAcc_{rhoAcc}, finAcc_{
                                                                      finAcc} {
    nx_ = params.nx;
    ny_ = params.ny;
    npop_ = params.npop;
  }

  void operator()(sycl::id<2> id) const {
    size_t j = id[0]; // along ny
    size_t i = id[1]; // along nx

    size_t index = i + nx_ * j;
    T cu = 0.0;
    T uX = uxAcc_[index];
    T usqr = 3.0 / 2 * (uX * uX);

    for (int ipop = 0; ipop < npop_; ++ipop) {
      cu = 3 * (vAcc_[ipop * 2] * uX);
      // along the for loop, data is not contiguous. Data is contiguous along
      // x, then nx-spaced along y, and nx*ny spaced along npop dimension
      finAcc_[index + ipop * nx_ * ny_] =
          rhoAcc_[index] * tAcc_[ipop] * (1 + cu + 0.5 * cu * cu - usqr);
    }
  }

private:
  syclReadAccessor uxAcc_;
  syclReadAccessor rhoAcc_;
  syclConstReadAccessor vAcc_;
  syclConstReadAccessor tAcc_;
  syclDiscardWriteAccessor finAcc_;

  size_t nx_;
  size_t ny_;
  size_t npop_;
};

template <typename T> struct Outflow {

  using syclReadWriteAccessor =
      sycl::accessor<T, 1, sycl::access::mode::read_write>;

public:
  Outflow(LBMParams<T> params, syclReadWriteAccessor finAcc)
      : nx_{params.nx}, nxny_{params.nx * params.ny}, finAcc_{finAcc} {}

  void operator()(sycl::id<1> j) const {
    int index1 = nx_ - 1 + nx_ * j;
    int index2 = nx_ - 2 + nx_ * j;

    finAcc_[index1 + 6 * nxny_] = finAcc_[index2 + 6 * nxny_];
    finAcc_[index1 + 7 * nxny_] = finAcc_[index2 + 7 * nxny_];
    finAcc_[index1 + 8 * nxny_] = finAcc_[index2 + 8 * nxny_];
  }

private:
  syclReadWriteAccessor finAcc_;
  int nxny_;
  int nx_;
};

template <typename T> struct Macroscopic {
  using syclDiscardWriteAccessor =
      sycl::accessor<T, 1, sycl::access::mode::discard_write>;

  using syclReadAccessor = sycl::accessor<T, 1, sycl::access::mode::read>;
  using syclConstReadAccessor =
      sycl::accessor<T, 1, sycl::access::mode::read,
                     sycl::access::target::constant_buffer>;

public:
  Macroscopic(LBMParams<T> params, syclReadAccessor finAcc,
              syclConstReadAccessor vAcc, syclDiscardWriteAccessor uxAcc,
              syclDiscardWriteAccessor uyAcc, syclDiscardWriteAccessor rhoAcc)
      : nx_{params.nx}, nxny_{params.nx * params.ny}, npop_{params.npop},
        finAcc_{finAcc}, vAcc_{vAcc}, uxAcc_{uxAcc}, uyAcc_{uyAcc},
        rhoAcc_{rhoAcc} {}

  void operator()(sycl::id<2> id) const {
    size_t i = id[1]; // along nx
    size_t j = id[0]; // along ny

    int index = i + nx_ * j;

    T rho_tmp = 0.0;
    T ux_tmp = 0.0;
    T uy_tmp = 0.0;
    T tempFin = 0.0;

    for (int ipop = 0; ipop < npop_; ++ipop) {

      tempFin = finAcc_[index + ipop * nxny_];

      // Oth order moment
      rho_tmp += tempFin;

      // 1st order moment
      ux_tmp += vAcc_[ipop * 2] * tempFin;
      uy_tmp += vAcc_[ipop * 2 + 1] * tempFin;

    } // end for ipop

    rhoAcc_[index] = rho_tmp;
    uxAcc_[index] = ux_tmp / rho_tmp;
    uyAcc_[index] = uy_tmp / rho_tmp;
  }

private:
  syclDiscardWriteAccessor uxAcc_;
  syclDiscardWriteAccessor uyAcc_;
  syclDiscardWriteAccessor rhoAcc_;
  syclReadAccessor finAcc_;
  syclConstReadAccessor vAcc_;

  int npop_;
  int nx_;
  int nxny_;
};

template <typename T> class InflowMacro {

  using syclReadAccessor = sycl::accessor<T, 1, sycl::access::mode::read>;
  using syclWriteAccessor = sycl::accessor<T, 1, sycl::access::mode::write>;
  using syclReadWriteAccessor =
      sycl::accessor<T, 1, sycl::access::mode::read_write>;

public:
  InflowMacro(LBMParams<T> params, syclReadAccessor finAcc,
              syclReadWriteAccessor uxAcc, syclWriteAccessor uyAcc,
              syclWriteAccessor rhoAcc)
      : nx_{params.nx}, ly_{params.ly}, nxny_{params.nx * params.ny},
        uLB_{params.uLB}, finAcc_{finAcc}, uxAcc_{uxAcc}, uyAcc_{uyAcc},
        rhoAcc_{rhoAcc} {}

  void operator()(sycl::id<1> j) const {
    int index = nx_ * j;
    // Compute velocity (for ux only, since uy = 0 everywhere)
    uxAcc_[index] = uLB_ * (1.0 + 1e-4 * sycl::sin((T)j / ly_ * 2 * M_PI));
    uyAcc_[index] = 0.f;
    rhoAcc_[index] =
        1 / (1 - uxAcc_[index]) *
        (finAcc_[index + 3 * nxny_] + finAcc_[index + 4 * nxny_] +
         finAcc_[index + 5 * nxny_] +
         2 * (finAcc_[index + 6 * nxny_] + finAcc_[index + 7 * nxny_] +
              finAcc_[index + 8 * nxny_]));
  }

private:
  syclReadAccessor finAcc_;
  syclWriteAccessor uyAcc_;
  syclWriteAccessor rhoAcc_;
  syclReadWriteAccessor uxAcc_;
  T uLB_;
  int nx_;
  T ly_;
  int nxny_;
};

template <typename T> class Equilibrium {

  using syclDiscardWriteAccessor =
      sycl::accessor<T, 1, sycl::access::mode::discard_write>;

  using syclReadAccessor = sycl::accessor<T, 1, sycl::access::mode::read>;
  using syclConstReadAccessor =
      sycl::accessor<T, 1, sycl::access::mode::read,
                     sycl::access::target::constant_buffer>;

public:
  Equilibrium(LBMParams<T> params, syclReadAccessor uxAcc, syclReadAccessor uyAcc,
              syclReadAccessor rhoAcc, syclConstReadAccessor vAcc,
              syclConstReadAccessor tAcc, syclDiscardWriteAccessor feqAcc)
      : nx_{params.nx}, nxny_{params.nx * params.ny}, npop_{params.npop},
        uxAcc_{uxAcc}, uyAcc_{uyAcc}, rhoAcc_{rhoAcc}, vAcc_{vAcc}, tAcc_{tAcc},
        feqAcc_{feqAcc} {}

  void operator()(sycl::id<2> id) const {

    size_t i = id[1]; // along nx
    size_t j = id[0]; // along ny

    int index = i + nx_ * j;

    T usqr =
        3.0 / 2 *
        (uxAcc_[index] * uxAcc_[index] + uyAcc_[index] * uyAcc_[index]);
    for (int ipop = 0; ipop < npop_; ++ipop) {
      T cu = 3 * (vAcc_[ipop * 2] * uxAcc_[index] +
                       vAcc_[ipop * 2 + 1] * uyAcc_[index]);

      feqAcc_[index + ipop * nxny_] =
          rhoAcc_[index] * tAcc_[ipop] * (1 + cu + 0.5 * cu * cu - usqr);
    }
  }

private:
  syclReadAccessor uxAcc_;
  syclReadAccessor uyAcc_;
  syclReadAccessor rhoAcc_;

  syclConstReadAccessor vAcc_;
  syclConstReadAccessor tAcc_;

  syclDiscardWriteAccessor feqAcc_;

  int nx_;
  int nxny_;
  int npop_;
};

template <typename T> class InflowDistr {

  using syclReadAccessor = sycl::accessor<T, 1, sycl::access::mode::read>;
  using syclReadWriteAccessor =
      sycl::accessor<T, 1, sycl::access::mode::read_write>;

public:
  InflowDistr(LBMParams<T> params, syclReadWriteAccessor finAcc,
              syclReadAccessor feqAcc)
      : nx_{params.nx}, nxny_{params.nx * params.ny}, finAcc_{finAcc},
        feqAcc_{feqAcc} {}

  void operator()(sycl::id<1> j) const {
    int index = nx_ * j;

    finAcc_[index + 0 * nxny_] = feqAcc_[index + 0 * nxny_] +
                                 finAcc_[index + 8 * nxny_] -
                                 feqAcc_[index + 8 * nxny_];
    finAcc_[index + 1 * nxny_] = feqAcc_[index + 1 * nxny_] +
                                 finAcc_[index + 7 * nxny_] -
                                 feqAcc_[index + 7 * nxny_];
    finAcc_[index + 2 * nxny_] = feqAcc_[index + 2 * nxny_] +
                                 finAcc_[index + 6 * nxny_] -
                                 feqAcc_[index + 6 * nxny_];
  }

private:
  syclReadAccessor feqAcc_;
  syclReadWriteAccessor finAcc_;

  int nx_;
  int nxny_;
};

template <typename T> class Collision {

  using syclReadAccessor = sycl::accessor<T, 1, sycl::access::mode::read>;
  using syclDiscardWriteAccessor =
      sycl::accessor<T, 1, sycl::access::mode::discard_write>;

public:
  Collision(LBMParams<T> params, syclReadAccessor finAcc, syclReadAccessor feqAcc,
            syclDiscardWriteAccessor foutAcc)
      : nx_{params.nx}, nxny_{params.nx * params.ny}, npop_{params.npop},
        omega_{params.omega}, finAcc_{finAcc}, feqAcc_{feqAcc}, foutAcc_{
                                                                    foutAcc} {}

  void operator()(sycl::id<2> id) const {
    size_t i = id[1]; // along nx
    size_t j = id[0]; // along ny

    int index = i + nx_ * j;
    int index_f = index - nxny_;

    for (int ipop = 0; ipop < npop_; ++ipop) {
      index_f += nxny_;
      foutAcc_[index_f] =
          finAcc_[index_f] - omega_ * (finAcc_[index_f] - feqAcc_[index_f]);
    } // end for ipop
  }

private:
  syclReadAccessor finAcc_;
  syclReadAccessor feqAcc_;
  syclDiscardWriteAccessor foutAcc_;

  int nx_;
  int nxny_;
  int npop_;
  T omega_;
};

template <typename T, typename I> class UpdateObstacle {

  using syclReadAccessor = sycl::accessor<T, 1, sycl::access::mode::read>;
  using syclIReadAccessor = sycl::accessor<I, 1, sycl::access::mode::read>;
  using syclWriteAccessor = sycl::accessor<T, 1, sycl::access::mode::write>;

public:
  UpdateObstacle(LBMParams<T> params, syclReadAccessor finAcc,
                 syclIReadAccessor obstAcc, syclWriteAccessor foutAcc)
      : nx_{params.nx}, nxny_{params.nx * params.ny}, npop_{params.npop},
        finAcc_{finAcc}, obstAcc_{obstAcc}, foutAcc_{foutAcc} {}

  void operator()(sycl::id<2> id) const {
    size_t i = id[1]; // along nx
    size_t j = id[0]; // along ny

    int index = i + nx_ * j;

    if (obstAcc_[index] == 1) {
      for (int ipop = 0; ipop < npop_; ++ipop) {

        int index_out = index + ipop * nxny_;
        int index_in = index + (8 - ipop) * nxny_;

        foutAcc_[index_out] = finAcc_[index_in];

      } // end for ipop
    }
  }

private:
  syclReadAccessor finAcc_;
  syclIReadAccessor obstAcc_;
  syclWriteAccessor foutAcc_;

  int nx_;
  int nxny_;
  int npop_;
};

template <typename T> class Streaming {

  using syclReadAccessor = sycl::accessor<T, 1, sycl::access::mode::read>;
  using syclDiscardWriteAccessor =
      sycl::accessor<T, 1, sycl::access::mode::discard_write>;
  using syclConstReadAccessor =
      sycl::accessor<T, 1, sycl::access::mode::read,
                     sycl::access::target::constant_buffer>;

public:
  Streaming(LBMParams<T> params, syclReadAccessor foutAcc,
            syclConstReadAccessor vAcc, syclDiscardWriteAccessor finAcc)
      : nx_{params.nx}, ny_{params.ny}, nxny_{params.nx * params.ny},
        npop_{params.npop}, foutAcc_{foutAcc}, vAcc_{vAcc}, finAcc_{finAcc} {}

  void operator()(sycl::id<2> id) const {
    size_t i = id[1]; // along nx
    size_t j = id[0]; // along ny

    int index = i + nx_ * j;

    int index_in, index_out, j_out, i_out = 0;

    for (int ipop = 0; ipop < npop_; ++ipop) {
      index_in = index + ipop * nxny_;
      i_out = i - vAcc_[2 * ipop];

      if (i_out < 0)
        i_out += nx_;
      if (i_out > nx_ - 1)
        i_out -= nx_;

      j_out = j - vAcc_[2 * ipop + 1];
      if (j_out < 0)
        j_out += ny_;
      if (j_out > ny_ - 1)
        j_out -= ny_;

      index_out = i_out + nx_ * j_out + ipop * nxny_;

      finAcc_[index_in] = foutAcc_[index_out];

    } // end for ipop
  }

private:
  syclReadAccessor foutAcc_;
  syclDiscardWriteAccessor finAcc_;
  syclConstReadAccessor vAcc_;

  int nx_;
  int ny_;
  int nxny_;
  int npop_;
};

template <typename T, typename I> class ImageCompute {
  using syclReadAccessor = sycl::accessor<T, 1, sycl::access::mode::read>;
  using syclDiscardWriteAccessor =
      sycl::accessor<I, 1, sycl::access::mode::discard_write>;

public:
  ImageCompute(LBMParams<T> params, syclReadAccessor u2Acc,
               syclReadAccessor maxAcc, syclReadAccessor minAcc,
               syclDiscardWriteAccessor imgAcc)
      : nx_{params.nx}, u2Acc_{u2Acc}, maxAcc_{maxAcc}, minAcc_{minAcc},
        imgAcc_{imgAcc} {}

  void operator()(sycl::id<2> id) const {

    size_t i = id[1]; // along nx
    size_t j = id[0]; // along ny

    int index = i + nx_ * j;

    // rescale velocity in 0-255 range
    I value = static_cast<I>((u2Acc_[index] - minAcc_[0]) /
                             (maxAcc_[0] - minAcc_[0]) * 255);
    imgAcc_[0 + 4 * i + 4 * nx_ * j] = 255 - value;
    imgAcc_[1 + 4 * i + 4 * nx_ * j] = value;
    imgAcc_[2 + 4 * i + 4 * nx_ * j] = 0;
    imgAcc_[3 + 4 * i + 4 * nx_ * j] = 255 - static_cast<I>(value / 2);
  }

private:
  syclReadAccessor u2Acc_;
  syclReadAccessor maxAcc_;
  syclReadAccessor minAcc_;
  syclDiscardWriteAccessor imgAcc_;

  int nx_;
};