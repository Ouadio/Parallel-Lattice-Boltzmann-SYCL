#include <cstdlib> // for malloc
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "LBMSolver.h"
#include "profiling/sycl_profiler_utils.h"
#include "writePNG/lodepng.h"
#include "writeVTK/saveVTK.h"

#include <CL/sycl.hpp>

using namespace cl;

// ======================================================
// ======================================================
template<typename T> LBMSolver<T>::LBMSolver(const LBMParams<T> &params, sycl::device device)
    : params(params), queue{device,
                            [](sycl::exception_list el) {
                              for (auto ex : el) {
                                std::rethrow_exception(ex);
                              }
                            }},
      lbm_vars{params} {} // LBMSolver::LBMSolver

// ======================================================
// ======================================================
template<typename T> void LBMSolver<T>::run() {

  // Simulation settings & parameters
  const size_t nx = params.nx;
  const size_t ny = params.ny;
  const int nxny = nx * ny;
  int maxIter = params.maxIter;
  int outStep = params.outStep;
  bool outImage = params.outImage;
  const size_t npop = params.npop;
  const T uLB = params.uLB;
  const T ly = params.ly;
  const T omega = params.omega;


  // -------------- INIT obstacle mask array

  queue.submit([&](sycl::handler &cgh) {
    auto obstAcc =
        lbm_vars.obstacleBuff.template get_access<sycl::access::mode::discard_write>(
            cgh);

    cgh.parallel_for(sycl::range<2>{ny, nx},
                     ObstacleKernel<T, uint8_t>(params, obstAcc));
  });

  // -------------- INIT MACRO
  // rho is one everywhere : fill 1
  const T fillRho = 1.0;

  queue.submit([&](sycl::handler &cgh) {
    auto rhoAcc =
        lbm_vars.rho.template get_access<sycl::access::mode::discard_write>(cgh);
    cgh.fill<T>(rhoAcc, fillRho);
  });

  // uy is zero everywhere (right here/now only): fill 0
  const T fillUy = 0.0;

  queue.submit([&](sycl::handler &cgh) {
    auto uyAcc = lbm_vars.uy.template get_access<sycl::access::mode::discard_write>(cgh);
    cgh.fill<T>(uyAcc, fillUy);
  });

  // fill ux

  queue.submit([&](sycl::handler &cgh) {
    auto uxAcc = lbm_vars.ux.template get_access<sycl::access::mode::discard_write>(cgh);

    cgh.parallel_for(sycl::range<2>{ny, nx}, [=](sycl::id<2> id) {
      size_t i = id[1]; // along nx
      size_t j = id[0]; // along ny

      size_t index = i + nx * j;

      uxAcc[index] = uLB * (1.0 + 1e-4 * sycl::sin((T)j / ly * 2 * M_PI));
    });
  });

  // -------------- INIT EQ
  queue.submit([&](sycl::handler &cgh) {
    auto uxAcc = lbm_vars.ux.template get_access<sycl::access::mode::read>(cgh);
    auto rhoAcc = lbm_vars.rho.template get_access<sycl::access::mode::read>(cgh);

    auto tAcc =
        lbm_vars.t.template get_access<sycl::access::mode::read, sycl::access::target::constant_buffer>(cgh);
    auto vAcc =
        lbm_vars.v.template get_access<sycl::access::mode::read, sycl::access::target::constant_buffer>(cgh);

    auto finAcc =
        lbm_vars.fin.template get_access<sycl::access::mode::discard_write>(cgh);

    cgh.parallel_for(
        sycl::range<2>{ny, nx},
        InitEquilibrium<T>(params, vAcc, tAcc, uxAcc, rhoAcc, finAcc));
  });

  // time loop
  for (int iTime = 0; iTime <= maxIter; ++iTime) {

    queue.submit([&](sycl::handler &cgh) {
      auto finAcc =
          lbm_vars.fin.template get_access<sycl::access::mode::read_write>(cgh);

      cgh.parallel_for(sycl::range<1>{ny}, Outflow<T>(params, finAcc));
    });

    // -------------- MACRO

    queue.submit([&](sycl::handler &cgh) {
      auto uxAcc =
          lbm_vars.ux.template get_access<sycl::access::mode::discard_write>(cgh);
      auto uyAcc =
          lbm_vars.uy.template get_access<sycl::access::mode::discard_write>(cgh);
      auto rhoAcc =
          lbm_vars.rho.template get_access<sycl::access::mode::discard_write>(cgh);

      auto vAcc =
          lbm_vars.v.template get_access<sycl::access::mode::read, sycl::access::target::constant_buffer>(cgh);

      auto finAcc = lbm_vars.fin.template get_access<sycl::access::mode::read>(cgh);

      cgh.parallel_for(
          sycl::range<2>{ny, nx},
          Macroscopic<T>(params, finAcc, vAcc, uxAcc, uyAcc, rhoAcc));
    });

    // -------------- Left wall: inflow condition.

    queue.submit([&](sycl::handler &cgh) {
      auto uxAcc = lbm_vars.ux.template get_access<sycl::access::mode::read_write>(cgh);
      auto uyAcc = lbm_vars.uy.template get_access<sycl::access::mode::write>(cgh);
      auto rhoAcc = lbm_vars.rho.template get_access<sycl::access::mode::write>(cgh);
      auto finAcc = lbm_vars.fin.template get_access<sycl::access::mode::read>(cgh);

      cgh.parallel_for(
          sycl::range<1>{ny},
          InflowMacro<T>(params, finAcc, uxAcc, uyAcc, rhoAcc));
    });

    // Output (Image / VTK)
    if (!(iTime % outStep)) {

      if (outImage) {

        queue.submit([&](sycl::handler &cgh) {
          auto uxAcc = lbm_vars.ux.template get_access<sycl::access::mode::read>(cgh);
          auto uyAcc = lbm_vars.uy.template get_access<sycl::access::mode::read>(cgh);
          auto u2Acc =
              lbm_vars.u2.template get_access<sycl::access::mode::discard_write>(cgh);

          cgh.parallel_for(
              sycl::nd_range<1>{ny * nx, 1},
              sycl::reduction(
                  lbm_vars.max, cgh, sycl::maximum<>(),
                  sycl::property::reduction::initialize_to_identity()),
              [=](sycl::nd_item<1> item, auto &max_value) {
                size_t index = item.get_global_id();

                T uX = uxAcc[index];
                T uY = uyAcc[index];

                T uu2 = sycl::sqrt(uX * uX + uY * uY);

                u2Acc[index] = uu2;

                max_value.combine(uu2);
              });
        });

        queue.submit([&](sycl::handler &cgh) {
          auto u2Acc = lbm_vars.u2.template get_access<sycl::access::mode::read>(cgh);

          cgh.parallel_for(
              sycl::nd_range<1>{ny * nx, 1},
              sycl::reduction(
                  lbm_vars.min, cgh, sycl::minimum<>(),
                  sycl::property::reduction::initialize_to_identity()),
              [=](sycl::nd_item<1> item, auto &min_value) {
                size_t index = item.get_global_id();

                min_value.combine(u2Acc[index]);
              });
        });

        queue.submit([&](sycl::handler &cgh) {
          auto u2Acc = lbm_vars.u2.template get_access<sycl::access::mode::read>(cgh);

          auto imgAcc =
              lbm_vars.img.template get_access<sycl::access::mode::discard_write>(cgh);

          auto maxAcc = lbm_vars.max.template get_access<sycl::access::mode::read>(cgh);
          auto minAcc = lbm_vars.min.template get_access<sycl::access::mode::read>(cgh);

          cgh.parallel_for(sycl::range<2>{ny, nx},
                           ImageCompute<T, unsigned char>(
                               params, u2Acc, maxAcc, minAcc, imgAcc));
        });

        queue.submit([&](sycl::handler &cgh) {
          auto imgAcc = lbm_vars.img.template get_access<sycl::access::mode::read>(cgh);
          cgh.copy(imgAcc, lbm_vars.imgH);
        });
        queue.wait_and_throw();

      }      // Image output case
      else { // VTK Output case
        queue.submit([&](sycl::handler &cgh) {
          auto uxAcc = lbm_vars.ux.template get_access<sycl::access::mode::read>(cgh);

          cgh.copy(uxAcc, lbm_vars.uxH);
        });

        queue.submit([&](sycl::handler &cgh) {
          auto uyAcc = lbm_vars.uy.template get_access<sycl::access::mode::read>(cgh);

          cgh.copy(uyAcc, lbm_vars.uyH);
        });

        queue.submit([&](sycl::handler &cgh) {
          auto rhoAcc = lbm_vars.rho.template get_access<sycl::access::mode::read>(cgh);
          cgh.copy(rhoAcc, lbm_vars.rhoH);
        });
      } // VTK output case
    }

    // Compute equilibrium.

    // --------------- EQUILIB

    queue.submit([&](sycl::handler &cgh) {
      auto uxAcc = lbm_vars.ux.template get_access<sycl::access::mode::read>(cgh);
      auto uyAcc = lbm_vars.uy.template get_access<sycl::access::mode::read>(cgh);
      auto rhoAcc = lbm_vars.rho.template get_access<sycl::access::mode::read>(cgh);

      auto tAcc =
          lbm_vars.t.template get_access<sycl::access::mode::read, sycl::access::target::constant_buffer>(cgh);
      auto vAcc =
          lbm_vars.v.template get_access<sycl::access::mode::read, sycl::access::target::constant_buffer>(cgh);

      auto feqAcc =
          lbm_vars.feq.template get_access<sycl::access::mode::discard_write>(cgh);

      cgh.parallel_for(sycl::range<2>{ny, nx},
                       Equilibrium<T>(params, uxAcc, uyAcc, rhoAcc, vAcc,
                                           tAcc, feqAcc));
    });

    // --------------- UPDATE INFLOW

    queue.submit([&](sycl::handler &cgh) {
      auto feqAcc = lbm_vars.feq.template get_access<sycl::access::mode::read>(cgh);
      auto finAcc =
          lbm_vars.fin.template get_access<sycl::access::mode::read_write>(cgh);

      cgh.parallel_for(sycl::range<1>{ny},
                       InflowDistr<T>(params, finAcc, feqAcc));
    });

    // --------------- COLLISION

    queue.submit([&](sycl::handler &cgh) {
      auto finAcc = lbm_vars.fin.template get_access<sycl::access::mode::read>(cgh);
      auto foutAcc =
          lbm_vars.fout.template get_access<sycl::access::mode::discard_write>(cgh);
      auto feqAcc = lbm_vars.feq.template get_access<sycl::access::mode::read>(cgh);

      cgh.parallel_for(sycl::range<2>{ny, nx},
                       Collision<T>(params, finAcc, feqAcc, foutAcc));
    });

    // --------------- UPDATE OBSTACLE

    queue.submit([&](sycl::handler &cgh) {
      auto obstAcc =
          lbm_vars.obstacleBuff.template get_access<sycl::access::mode::read>(cgh);
      auto finAcc = lbm_vars.fin.template get_access<sycl::access::mode::read>(cgh);
      auto foutAcc = lbm_vars.fout.template get_access<sycl::access::mode::write>(cgh);

      cgh.parallel_for(
          sycl::range<2>{ny, nx},
          UpdateObstacle<T, uint8_t>(params, finAcc, obstAcc, foutAcc));
    });

    // --------------- STREAMING

    queue.submit([&](sycl::handler &cgh) {
      auto vAcc =
          lbm_vars.v.template get_access<sycl::access::mode::read, sycl::access::target::constant_buffer>(cgh);

      auto foutAcc = lbm_vars.fout.template get_access<sycl::access::mode::read>(cgh);
      auto finAcc =
          lbm_vars.fin.template get_access<sycl::access::mode::discard_write>(cgh);

      cgh.parallel_for(sycl::range<2>{ny, nx},
                       Streaming<T>(params, foutAcc, vAcc, finAcc));
    });

    if (!(iTime % outStep)) {
      if (outImage) {
#ifdef PROFILE
        COMPUTE_ELAPSED_TIME(prepare_png_output, true, elapsedTime)
#endif

        std::cout << "Output data (PNG) at time " << iTime << "\n";

        // create png image buff
        std::vector<unsigned char> image;
        image.resize(nx * ny * 4);
        image.assign(lbm_vars.imgH, lbm_vars.imgH + (nx * ny * 4));

        std::ostringstream iTimeNum;
        iTimeNum.width(7);
        iTimeNum.fill('0');
        iTimeNum << iTime;

        std::string filename = "vel_gpu_" + iTimeNum.str() + ".png";

        // encode the image
        unsigned error = lodepng::encode(filename, image, nx, ny);

        // if there's an error, display it
        if (error)
          std::cout << "encoder error " << error << ": "
                    << lodepng_error_text(error) << std::endl;
      } // Image output case
      else {
        output_vtk(iTime);
      } // VTK output case
    }

#ifdef PROFILE
    // PROFILING Repeated Routines

    COMPUTE_ELAPSED_TIME(border_outflow, true, elapsedTime)
    COMPUTE_ELAPSED_TIME(macroscopic, true, elapsedTime)
    COMPUTE_ELAPSED_TIME(border_inflow, true, elapsedTime)
    COMPUTE_ELAPSED_TIME(equilibrium, true, elapsedTime)
    COMPUTE_ELAPSED_TIME(update_fin_inflow, true, elapsedTime)
    COMPUTE_ELAPSED_TIME(compute_collision, true, elapsedTime)
    COMPUTE_ELAPSED_TIME(update_obstacle, true, elapsedTime)
    COMPUTE_ELAPSED_TIME(streaming, true, elapsedTime)
#endif

    // sycl::free(max_value, queue);
    // sycl::free(min_value, queue);
  } // end for iTime

  // delete[] imgH;
#ifdef PROFILE
  // PROFILING One Execution Routines

  COMPUTE_ELAPSED_TIME(init_obstacle_mask, false, elapsedTime)
  COMPUTE_ELAPSED_TIME(initialize_equilibrium, false, elapsedTime)
  COMPUTE_ELAPSED_TIME(initialize_macroscopic_variables, false, elapsedTime)
  COMPUTE_ELAPSED_TIME(copy_v, false, elapsedTime);
  COMPUTE_ELAPSED_TIME(copy_t, false, elapsedTime);
#endif

} // LBMSolver::run

// ======================================================
// ======================================================
template<typename T> void LBMSolver<T>::output_vtk(int iTime) {

  std::cout << "Output data (VTK) at time " << iTime << "\n";

  bool useAscii = false; // binary data leads to smaller files
  saveVTK<T>(lbm_vars.rhoH, lbm_vars.uxH, lbm_vars.uyH, params, useAscii, iTime);

} // LBMSolver::output_vtk

template<typename T> void LBMSolver<T>::printQueueInfo() {
  std::cout << "----------------------------------------" << std::endl;
  std::cout << "device's name : "
            << queue.get_device().get_info<sycl::info::device::name>()
            << std::endl;
  std::cout << "device's vendor : "
            << queue.get_device().get_info<sycl::info::device::vendor>()
            << std::endl;
  std::cout << "----------------------------------------\n" << std::endl;
}


template class LBMSolver<float>;
template class LBMSolver<double>;