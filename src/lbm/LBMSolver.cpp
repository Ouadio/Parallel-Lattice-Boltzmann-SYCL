#include <cstdlib> // for malloc
#include <iostream>
#include <vector>
#include <sstream>
#include <map>
#include <string>

#include "LBMSolver.h"

#include "lbmFlowUtils.h"

#include "writePNG/lodepng.h"
#include "writeVTK/saveVTK.h"

#include <CL/sycl.hpp>

using namespace cl;

// ======================================================
// ======================================================
LBMSolver::LBMSolver(const LBMParams &params) : params(params),
#ifdef SYCL_GPU
                                                queue
{
  sycl::gpu_selector{}, sycl::property::queue::enable_profiling {}
}
#else
                                                queue
{
  sycl::cpu_selector{}, sycl::property::queue::enable_profiling {}
}
#endif
{
  const int nx = params.nx;
  const int ny = params.ny;
  const int npop = LBMParams::npop;

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
  this->img = sycl::malloc_shared<unsigned char>(nx * ny * 4, queue);

  // obstacle
  this->obstacle = sycl::malloc_device<uint8_t>(nx * ny, queue);

  // weights and velocity
  this->t = sycl::malloc_device<real_t>(npop, queue);
  this->v = sycl::malloc_device<real_t>(2 * npop, queue);

  // Reserve memory for some vector events
  initialize_macroscopic_variables_ev.reserve(3); // 3 kernels within initialize_macroscopic_variables
  prepare_png_output_ev.reserve(3);

} // LBMSolver::LBMSolver

// ======================================================
// ======================================================
LBMSolver::~LBMSolver()
{
  const int nx = params.nx;
  const int ny = params.ny;
  const int npop = LBMParams::npop;

  // free memory

  // distribution functions
  sycl::free(fin, queue);
  sycl::free(fout, queue);
  sycl::free(feq, queue);

  // macroscopic variables

  sycl::free(rho, queue);
  sycl::free(ux, queue);
  sycl::free(uy, queue);

  // image output
  sycl::free(u2, queue);
  sycl::free(img, queue);

  // obstacle
  sycl::free(obstacle, queue);

  // weights and velocity
  sycl::free(t, queue);
  sycl::free(v, queue);

  // free allocated duplicates of rho, ux and uy (host side) if vtk output used
  if (!params.outImage)
  {
    delete[] uxH;
    delete[] uyH;
    delete[] rhoH;
  }

} // LBMSolver::~LBMSolver

// ======================================================
// ======================================================
void LBMSolver::initialize()
{

  // initialize obstacle mask array
  init_obstacle_mask(params,
                     obstacle,
                     queue,
                     init_obstacle_mask_ev);

  // anticipate mem-copy of weight and velocity vector for future kernel (initialize_equilibrium)
  // enables some data transfer & computation overlapping
  copy_v_ev = queue.submit([&](sycl::handler &cgh)
                           { cgh.memcpy(v, vHost, 2 * params.npop * sizeof(real_t)); });

  copy_t_ev = queue.submit([&](sycl::handler &cgh)
                           { cgh.memcpy(t, tHost, params.npop * sizeof(real_t)); });

  // initialize macroscopic velocity
  initialize_macroscopic_variables(params,
                                   rho,
                                   ux,
                                   uy,
                                   queue,
                                   initialize_macroscopic_variables_ev);

  // Initialization of the populations at equilibrium
  // with the given macroscopic variables.
  initialize_equilibrium(params,
                         v,
                         t,
                         rho,
                         ux,
                         uy,
                         fin,
                         queue,
                         initialize_equilibrium_ev);

} // LBMSolver::initialize

// ======================================================
// ======================================================
void LBMSolver::run()
{
  const int nx = params.nx;
  const int ny = params.ny;
  int maxIter = params.maxIter;
  int outStep = params.outStep;
  bool outImage = params.outImage;

  if (!outImage)
  {
    rhoH = new real_t[nx * ny];
    uxH = new real_t[nx * ny];
    uyH = new real_t[nx * ny];
  }

  // temporary variables for reduction purposes (image generation)

  real_t *max_value = sycl::malloc_shared<real_t>(1, queue);
  real_t *min_value = sycl::malloc_shared<real_t>(1, queue);

  // Initialize Simulation Variables (Obstacle, Macro variables, copy relevant data H2D )
  this->initialize();

  // time loop
  for (int iTime = 0; iTime <= maxIter; ++iTime)
  {

    // Right wall: outflow condition.
    // we only need here to specify distrib. function for velocities
    // that enter the domain (other that go out, are set by the streaming step)

    border_outflow(params, fin,
                   queue,
                   border_outflow_ev);
    if (!iTime)
    { // First iteration
      border_outflow_ev.wait({initialize_equilibrium_ev});
    }
    else
    { // >1 Iteration number
      border_outflow_ev.wait({streaming_ev});
    }

    // Compute macroscopic variables, density and velocity.

    macroscopic(params,
                v,
                fin,
                rho,
                ux,
                uy,
                queue,
                macroscopic_ev);

    macroscopic_ev.wait({border_outflow_ev});

    // Left wall: inflow condition.

    border_inflow(params,
                  fin,
                  rho,
                  ux,
                  uy,
                  queue,
                  border_inflow_ev);

    border_inflow_ev.wait({macroscopic_ev});

    if (!(iTime % outStep))
    {
      if (outImage)
      {
        prepare_png_output(params,
                           ux,
                           uy,
                           u2,
                           img,
                           max_value,
                           min_value,
                           queue,
                           prepare_png_output_ev);
        prepare_png_output_ev.at(0).wait({border_inflow_ev});
        prepare_png_output_ev.at(1).wait({border_inflow_ev});

      } // Image output case
      else
      {
        prepare_png_output_ev.emplace_back(queue.memcpy(uxH, ux, nx * ny * sizeof(real_t)));
        prepare_png_output_ev.emplace_back(queue.memcpy(uyH, uy, nx * ny * sizeof(real_t)));
        prepare_png_output_ev.emplace_back(queue.memcpy(uyH, uy, nx * ny * sizeof(real_t)));
      } // VTK output case
    }

    // Compute equilibrium.

    equilibrium(params,
                v,
                t,
                rho,
                ux,
                uy,
                feq,
                queue,
                equilibrium_ev);

    equilibrium_ev.wait({border_inflow_ev});

    update_fin_inflow(params,
                      feq,
                      fin,
                      queue,
                      update_fin_inflow_ev);

    update_fin_inflow_ev.wait({equilibrium_ev});

    // Collision step.
    compute_collision(params,
                      fin,
                      feq,
                      fout,
                      queue,
                      compute_collision_ev);

    compute_collision_ev.wait({update_fin_inflow_ev});

    // Bounce-back condition for obstacle.
    update_obstacle(params,
                    fin,
                    obstacle,
                    fout,
                    queue,
                    update_obstacle_ev);

    update_obstacle_ev.wait({compute_collision_ev});

    // Streaming step.

    streaming(params,
              v,
              fout,
              fin,
              queue,
              streaming_ev);

    streaming_ev.wait({update_obstacle_ev});

    queue.wait();

    if (!(iTime % outStep))
    {
      if (outImage)
      {
#ifdef PROFILE
        COMPUTE_ELAPSED_TIME(prepare_png_output, true, elapsedTime)
#endif
        output_png_parallel(iTime, nx, ny, img);
      } // Image output case
      else
      {
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

  } // end for iTime

#ifdef PROFILE
  // PROFILING One Execution Routines

  COMPUTE_ELAPSED_TIME(init_obstacle_mask, false, elapsedTime)
  COMPUTE_ELAPSED_TIME(initialize_equilibrium, false, elapsedTime)
  COMPUTE_ELAPSED_TIME(initialize_macroscopic_variables, false, elapsedTime)
  COMPUTE_ELAPSED_TIME(copy_v, false, elapsedTime);
  COMPUTE_ELAPSED_TIME(copy_t, false, elapsedTime);
#endif

  sycl::free(max_value, queue);
  sycl::free(min_value, queue);

} // LBMSolver::run

// ======================================================
// ======================================================
void LBMSolver::output_png(int iTime)
{

  std::cout << "Output data (PNG) at time " << iTime << "\n";

  const int nx = params.nx;
  const int ny = params.ny;

  real_t *u2 = (real_t *)malloc(nx * ny * sizeof(real_t));

  // compute velocity norm, as well as min and max values
  real_t min_value = sqrt(ux[0] * ux[0] + uy[0] * uy[0]);
  real_t max_value = min_value;
  for (int j = 0; j < ny; ++j)
  {
    for (int i = 0; i < nx; ++i)
    {

      int index = i + nx * j;

      u2[index] = sqrt(ux[index] * ux[index] + uy[index] * uy[index]);

      if (u2[index] < min_value)
        min_value = u2[index];

      if (u2[index] > max_value)
        max_value = u2[index];

    } // end for i

  } // end for j

  // create png image buff
  std::vector<unsigned char> image;
  image.resize(nx * ny * 4);
  for (int j = 0; j < ny; ++j)
  {
    for (int i = 0; i < nx; ++i)
    {

      int index = i + nx * j;

      // rescale velocity in 0-255 range
      unsigned char value = static_cast<unsigned char>((u2[index] - min_value) / (max_value - min_value) * 255);
      image[0 + 4 * i + 4 * nx * j] = value;
      image[1 + 4 * i + 4 * nx * j] = value;
      image[2 + 4 * i + 4 * nx * j] = value;
      image[3 + 4 * i + 4 * nx * j] = value;
    }
  }

  std::ostringstream iTimeNum;
  iTimeNum.width(7);
  iTimeNum.fill('0');
  iTimeNum << iTime;

  std::string filename = "vel_test_" + iTimeNum.str() + ".png";

  // encode the image
  unsigned error = lodepng::encode(filename, image, nx, ny);

  //if there's an error, display it
  if (error)
    std::cout << "encoder error " << error << ": " << lodepng_error_text(error) << std::endl;

  delete[] u2;

} // LBMSolver::output_png

// ======================================================
// ======================================================
void LBMSolver::output_vtk(int iTime)
{

  std::cout << "Output data (VTK) at time " << iTime << "\n";

  bool useAscii = false; // binary data leads to smaller files
  saveVTK(rhoH, uxH, uyH, params, useAscii, iTime);

} // LBMSolver::output_vtk

void LBMSolver::printQueueInfo()
{
  std::cout << "----------------------------------------" << std::endl;
  std::cout << "device's name : " << queue.get_device().get_info<sycl::info::device::name>() << std::endl;
  std::cout << "device's vendor : " << queue.get_device().get_info<sycl::info::device::vendor>() << std::endl;
  std::cout << "----------------------------------------\n"
            << std::endl;
}