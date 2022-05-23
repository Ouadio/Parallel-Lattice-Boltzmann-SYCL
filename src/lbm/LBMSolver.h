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

// Accessors targets (global_buffer deprecated in DPCPP SYCL 2020)
#ifdef COMPUTECPP
#define DEVICE_TARGET sycl::access::target::global_buffer
#else
#define DEVICE_TARGET sycl::access::target::device
#endif

// Accessors
#define SyclReadWriteAccessor(T)                                               \
  sycl::accessor<T, 1, sycl::access::mode::read_write, DEVICE_TARGET>
#define SyclDiscardReadWriteAccessor(T)                                        \
  sycl::accessor<T, 1, sycl::access::mode::discard_read_write, DEVICE_TARGET>
#define SyclDiscardWriteAccessor(T)                                            \
  sycl::accessor<T, 1, sycl::access::mode::discard_write, DEVICE_TARGET>
#define SyclWriteAccessor(T)                                                   \
  sycl::accessor<T, 1, sycl::access::mode::write, DEVICE_TARGET>
#define SyclReadAccessor(T)                                                    \
  sycl::accessor<T, 1, sycl::access::mode::read, DEVICE_TARGET>

// Wrapper struct for LBM variables (D2Q9)
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

  sycl::buffer<T, 1> fout{sycl::range<1>{1}};

  sycl::buffer<T, 1> fin{sycl::range<1>{1}};
  sycl::buffer<T, 1> feq{sycl::range<1>{1}};

  // Constant buffers

  sycl::buffer<T, 1> t{sycl::range<1>{1}};
  sycl::buffer<T, 1> v{sycl::range<1>{1}};

  // Variables Buffers
  sycl::buffer<uint8_t, 1> obstacleBuff{sycl::range<1>{1}};

  sycl::buffer<T, 1> rho{sycl::range<1>{1}};
  sycl::buffer<T, 1> ux{sycl::range<1>{1}};
  sycl::buffer<T, 1> uy{sycl::range<1>{1}};

  sycl::buffer<T, 1> u2{sycl::range<1>{1}};
  sycl::buffer<unsigned char, 1> img{sycl::range<1>{1}};
  sycl::buffer<T, 1> max{sycl::range<1>{1}};
  sycl::buffer<T, 1> min{sycl::range<1>{1}};

  unsigned char *imgH{nullptr};

  bool outImage;

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
template <typename T> class LBMSolver {

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

public:
  ObstacleKernel(const LBMParams<T> &params,
                 SyclDiscardWriteAccessor(I) obstacleAcc)
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
  SyclDiscardWriteAccessor(I) obsAcc_;
  T nx_;
  T cx_;
  T cy_;
  T r2_;
};

template <typename T> struct InitEquilibrium {

  using syclConstReadAccessor =
      sycl::accessor<T, 1, sycl::access::mode::read,
                     sycl::access::target::constant_buffer>;

public:
  InitEquilibrium(const LBMParams<T> &params, syclConstReadAccessor vAcc,
                  syclConstReadAccessor tAcc, SyclReadAccessor(T) uxAcc,
                  SyclReadAccessor(T) rhoAcc,
                  SyclDiscardWriteAccessor(T) finAcc)
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
  SyclReadAccessor(T) uxAcc_;
  SyclReadAccessor(T) rhoAcc_;
  syclConstReadAccessor vAcc_;
  syclConstReadAccessor tAcc_;
  SyclDiscardWriteAccessor(T) finAcc_;

  size_t nx_;
  size_t ny_;
  size_t npop_;
};

template <typename T> struct Outflow {

public:
  Outflow(LBMParams<T> params, SyclReadWriteAccessor(T) finAcc)
      : nx_{params.nx}, nxny_{params.nx * params.ny}, finAcc_{finAcc} {}

  void operator()(sycl::id<1> j) const {
    int index1 = nx_ - 1 + nx_ * j.get(0);
    int index2 = nx_ - 2 + nx_ * j.get(0);

    finAcc_[index1 + 6 * nxny_] = finAcc_[index2 + 6 * nxny_];
    finAcc_[index1 + 7 * nxny_] = finAcc_[index2 + 7 * nxny_];
    finAcc_[index1 + 8 * nxny_] = finAcc_[index2 + 8 * nxny_];
  }

private:
  SyclReadWriteAccessor(T) finAcc_;
  int nxny_;
  int nx_;
};

template <typename T> struct Macroscopic {

  using syclConstReadAccessor =
      sycl::accessor<T, 1, sycl::access::mode::read,
                     sycl::access::target::constant_buffer>;

public:
  Macroscopic(LBMParams<T> params, SyclReadAccessor(T) finAcc,
              syclConstReadAccessor vAcc, SyclDiscardWriteAccessor(T) uxAcc,
              SyclDiscardWriteAccessor(T) uyAcc,
              SyclDiscardWriteAccessor(T) rhoAcc)
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
  SyclDiscardWriteAccessor(T) uxAcc_;
  SyclDiscardWriteAccessor(T) uyAcc_;
  SyclDiscardWriteAccessor(T) rhoAcc_;
  SyclReadAccessor(T) finAcc_;
  syclConstReadAccessor vAcc_;

  int npop_;
  int nx_;
  int nxny_;
};

template <typename T> class InflowMacro {

public:
  InflowMacro(LBMParams<T> params, SyclReadAccessor(T) finAcc,
              SyclReadWriteAccessor(T) uxAcc, SyclWriteAccessor(T) uyAcc,
              SyclWriteAccessor(T) rhoAcc)
      : nx_{params.nx}, ly_{params.ly}, nxny_{params.nx * params.ny},
        uLB_{params.uLB}, finAcc_{finAcc}, uxAcc_{uxAcc}, uyAcc_{uyAcc},
        rhoAcc_{rhoAcc} {}

  void operator()(sycl::id<1> j) const {
    int index = nx_ * j.get(0);
    // Compute velocity (for ux only, since uy = 0 everywhere)
    uxAcc_[index] =
        uLB_ * (1.0 + 1e-4 * sycl::sin((T)j.get(0) / ly_ * 2 * M_PI));
    uyAcc_[index] = 0.f;
    rhoAcc_[index] =
        1 / (1 - uxAcc_[index]) *
        (finAcc_[index + 3 * nxny_] + finAcc_[index + 4 * nxny_] +
         finAcc_[index + 5 * nxny_] +
         2 * (finAcc_[index + 6 * nxny_] + finAcc_[index + 7 * nxny_] +
              finAcc_[index + 8 * nxny_]));
  }

private:
  SyclReadAccessor(T) finAcc_;
  SyclWriteAccessor(T) uyAcc_;
  SyclWriteAccessor(T) rhoAcc_;
  SyclReadWriteAccessor(T) uxAcc_;
  T uLB_;
  int nx_;
  T ly_;
  int nxny_;
};

template <typename T> class Equilibrium {

  using syclConstReadAccessor =
      sycl::accessor<T, 1, sycl::access::mode::read,
                     sycl::access::target::constant_buffer>;

public:
  Equilibrium(LBMParams<T> params, SyclReadAccessor(T) uxAcc,
              SyclReadAccessor(T) uyAcc, SyclReadAccessor(T) rhoAcc,
              syclConstReadAccessor vAcc, syclConstReadAccessor tAcc,
              SyclDiscardWriteAccessor(T) feqAcc)
      : nx_{params.nx}, nxny_{params.nx * params.ny}, npop_{params.npop},
        uxAcc_{uxAcc}, uyAcc_{uyAcc}, rhoAcc_{rhoAcc}, vAcc_{vAcc}, tAcc_{tAcc},
        feqAcc_{feqAcc} {}

  void operator()(sycl::id<2> id) const {

    size_t i = id[1]; // along nx
    size_t j = id[0]; // along ny

    int index = i + nx_ * j;

    T usqr = 3.0 / 2 *
             (uxAcc_[index] * uxAcc_[index] + uyAcc_[index] * uyAcc_[index]);
    for (int ipop = 0; ipop < npop_; ++ipop) {
      T cu = 3 * (vAcc_[ipop * 2] * uxAcc_[index] +
                  vAcc_[ipop * 2 + 1] * uyAcc_[index]);

      feqAcc_[index + ipop * nxny_] =
          rhoAcc_[index] * tAcc_[ipop] * (1 + cu + 0.5 * cu * cu - usqr);
    }
  }

private:
  SyclReadAccessor(T) uxAcc_;
  SyclReadAccessor(T) uyAcc_;
  SyclReadAccessor(T) rhoAcc_;

  syclConstReadAccessor vAcc_;
  syclConstReadAccessor tAcc_;

  SyclDiscardWriteAccessor(T) feqAcc_;

  int nx_;
  int nxny_;
  int npop_;
};

template <typename T> class InflowDistr {

public:
  InflowDistr(LBMParams<T> params, SyclReadWriteAccessor(T) finAcc,
              SyclReadAccessor(T) feqAcc)
      : nx_{params.nx}, nxny_{params.nx * params.ny}, finAcc_{finAcc},
        feqAcc_{feqAcc} {}

  void operator()(sycl::id<1> j) const {
    int index = nx_ * j.get(0);

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
  SyclReadAccessor(T) feqAcc_;
  SyclReadWriteAccessor(T) finAcc_;

  int nx_;
  int nxny_;
};

template <typename T> class Collision {

public:
  Collision(LBMParams<T> params, SyclReadAccessor(T) finAcc,
            SyclReadAccessor(T) feqAcc, SyclDiscardWriteAccessor(T) foutAcc)
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
  SyclReadAccessor(T) finAcc_;
  SyclReadAccessor(T) feqAcc_;
  SyclDiscardWriteAccessor(T) foutAcc_;

  int nx_;
  int nxny_;
  int npop_;
  T omega_;
};

template <typename T, typename I> class UpdateObstacle {

public:
  UpdateObstacle(LBMParams<T> params, SyclReadAccessor(T) finAcc,
                 SyclReadAccessor(I) obstAcc, SyclWriteAccessor(T) foutAcc)
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
  SyclReadAccessor(T) finAcc_;
  SyclReadAccessor(I) obstAcc_;
  SyclWriteAccessor(T) foutAcc_;

  int nx_;
  int nxny_;
  int npop_;
};

template <typename T> class Streaming {

  using syclConstReadAccessor =
      sycl::accessor<T, 1, sycl::access::mode::read,
                     sycl::access::target::constant_buffer>;

public:
  Streaming(LBMParams<T> params, SyclReadAccessor(T) foutAcc,
            syclConstReadAccessor vAcc, SyclDiscardWriteAccessor(T) finAcc)
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
  SyclReadAccessor(T) foutAcc_;
  SyclDiscardWriteAccessor(T) finAcc_;
  syclConstReadAccessor vAcc_;

  int nx_;
  int ny_;
  int nxny_;
  int npop_;
};

template <typename T, typename I> class ImageCompute {

public:
  ImageCompute(LBMParams<T> params, SyclReadAccessor(T) u2Acc,
               SyclReadAccessor(T) maxAcc, SyclReadAccessor(T) minAcc,
               SyclDiscardWriteAccessor(I) imgAcc)
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
  SyclReadAccessor(T) u2Acc_;
  SyclReadAccessor(T) maxAcc_;
  SyclReadAccessor(T) minAcc_;
  SyclDiscardWriteAccessor(I) imgAcc_;

  int nx_;
};

template <typename T> class ReduceMaxMinFirst {

  using syclLocalReadWriteAccessor =
      sycl::accessor<T, 1, sycl::access::mode::read_write,
                     sycl::access::target::local>;

public:
  ReduceMaxMinFirst(SyclReadAccessor(T) u2Acc,
                    SyclDiscardWriteAccessor(T) writeRedMaxAcc,
                    SyclDiscardWriteAccessor(T) writeRedMinAcc,
                    SyclDiscardWriteAccessor(T) minAcc,
                    SyclDiscardWriteAccessor(T) maxAcc,
                    syclLocalReadWriteAccessor localMaxAcc,
                    syclLocalReadWriteAccessor localMinAcc, int group_size,
                    int length)
      : u2Acc_{u2Acc}, redMaxAcc_{writeRedMaxAcc}, redMinAcc_{writeRedMinAcc},
        minAcc_{minAcc}, maxAcc_{maxAcc}, localMaxAcc_{localMaxAcc},
        localMinAcc_{localMinAcc}, group_size_{group_size}, length_{length} {}

  void operator()(sycl::nd_item<1> item) {
    int local_id = item.get_local_linear_id();
    int global_id = item.get_global_linear_id();

    localMaxAcc_[local_id] = std::numeric_limits<T>::min();
    localMinAcc_[local_id] = std::numeric_limits<T>::max();

    if (2 * global_id < length_) {
      if (u2Acc_[2 * global_id] < u2Acc_[2 * global_id + 1]) {
        localMaxAcc_[local_id] = u2Acc_[2 * global_id + 1];
        localMinAcc_[local_id] = u2Acc_[2 * global_id];
      } else {
        localMaxAcc_[local_id] = u2Acc_[2 * global_id];
        localMinAcc_[local_id] = u2Acc_[2 * global_id + 1];
      }
    }

    item.barrier(sycl::access::fence_space::local_space);

    for (int stride = 1; stride < group_size_; stride *= 2) {
      auto idx = 2 * stride * local_id;
      if (idx < group_size_) {
        localMaxAcc_[idx] = localMaxAcc_[idx] < localMaxAcc_[idx + stride]
                                ? localMaxAcc_[idx + stride]
                                : localMaxAcc_[idx];
        localMinAcc_[idx] = localMinAcc_[idx] < localMinAcc_[idx + stride]
                                ? localMinAcc_[idx]
                                : localMinAcc_[idx + stride];
      }
      item.barrier(sycl::access::fence_space::local_space);
    }

    if (local_id == 0) {
      redMaxAcc_[item.get_group_linear_id()] = localMaxAcc_[0];
      redMinAcc_[item.get_group_linear_id()] = localMinAcc_[0];
      minAcc_[0] = localMinAcc_[0];
      maxAcc_[0] = localMaxAcc_[0];
    }
  }

private:
  SyclReadAccessor(T) u2Acc_;
  SyclDiscardWriteAccessor(T) redMaxAcc_;
  SyclDiscardWriteAccessor(T) redMinAcc_;
  SyclDiscardWriteAccessor(T) minAcc_;
  SyclDiscardWriteAccessor(T) maxAcc_;
  syclLocalReadWriteAccessor localMaxAcc_;
  syclLocalReadWriteAccessor localMinAcc_;
  int group_size_;
  int length_;
};

template <typename T> class ReduceMaxMin {

  using syclLocalReadWriteAccessor =
      sycl::accessor<T, 1, sycl::access::mode::read_write,
                     sycl::access::target::local>;

public:
  ReduceMaxMin(SyclDiscardReadWriteAccessor(T) redMaxAcc,
               SyclDiscardReadWriteAccessor(T) redMinAcc,
               SyclDiscardWriteAccessor(T) minAcc,
               SyclDiscardWriteAccessor(T) maxAcc,
               syclLocalReadWriteAccessor localMaxAcc,
               syclLocalReadWriteAccessor localMinAcc, int group_size,
               int length)
      : redMaxAcc_{redMaxAcc}, redMinAcc_{redMinAcc}, minAcc_{minAcc},
        maxAcc_{maxAcc}, localMaxAcc_{localMaxAcc}, localMinAcc_{localMinAcc},
        group_size_{group_size}, length_{length} {}

  void operator()(sycl::nd_item<1> item) {
    int local_id = item.get_local_linear_id();
    int global_id = item.get_global_linear_id();

    localMaxAcc_[local_id] = std::numeric_limits<T>::min();
    localMinAcc_[local_id] = std::numeric_limits<T>::max();

    if (2 * global_id < length_) {
      localMaxAcc_[local_id] =
          redMaxAcc_[2 * global_id] < redMaxAcc_[2 * global_id + 1]
              ? redMaxAcc_[2 * global_id + 1]
              : redMaxAcc_[2 * global_id];

      localMinAcc_[local_id] =
          redMinAcc_[2 * global_id] < redMinAcc_[2 * global_id + 1]
              ? redMinAcc_[2 * global_id]
              : redMinAcc_[2 * global_id + 1];
    }

    item.barrier(sycl::access::fence_space::local_space);

    for (int stride = 1; stride < group_size_; stride *= 2) {
      auto idx = 2 * stride * local_id;
      if (idx < group_size_) {
        localMaxAcc_[idx] = localMaxAcc_[idx] < localMaxAcc_[idx + stride]
                                ? localMaxAcc_[idx + stride]
                                : localMaxAcc_[idx];
        localMinAcc_[idx] = localMinAcc_[idx] < localMinAcc_[idx + stride]
                                ? localMinAcc_[idx]
                                : localMinAcc_[idx + stride];
      }
      item.barrier(sycl::access::fence_space::local_space);
    }

    if (local_id == 0) {
      redMaxAcc_[item.get_group_linear_id()] = localMaxAcc_[0];
      redMinAcc_[item.get_group_linear_id()] = localMinAcc_[0];
      minAcc_[0] = localMinAcc_[0];
      maxAcc_[0] = localMaxAcc_[0];
    }
  }

private:
  SyclDiscardReadWriteAccessor(T) redMaxAcc_;
  SyclDiscardReadWriteAccessor(T) redMinAcc_;
  SyclDiscardWriteAccessor(T) minAcc_;
  SyclDiscardWriteAccessor(T) maxAcc_;
  syclLocalReadWriteAccessor localMaxAcc_;
  syclLocalReadWriteAccessor localMinAcc_;
  int group_size_;
  int length_;
};

template <typename T> class SpeedCompute {

  using syclWriteAccessor =
      sycl::accessor<T, 1, sycl::access::mode::discard_write, DEVICE_TARGET>;

public:
  SpeedCompute(SyclReadAccessor(T) uxAcc, SyclReadAccessor(T) uyAcc,
               syclWriteAccessor u2Acc)
      : uxAcc_{uxAcc}, uyAcc_{uyAcc}, u2Acc_{u2Acc} {}

  void operator()(sycl::id<1> id) {
    int index = id.get(0);

    T uX = uxAcc_[index];
    T uY = uyAcc_[index];

    u2Acc_[index] = sycl::sqrt<T>(uX * uX + uY * uY);
  }

private:
  syclWriteAccessor u2Acc_;
  SyclReadAccessor(T) uxAcc_;
  SyclReadAccessor(T) uyAcc_;
};