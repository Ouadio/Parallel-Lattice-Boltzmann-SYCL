#include <math.h> // for M_PI = 3.1415....

#include "lbmFlowUtils.h"
#include <iostream>
#include "writePNG/lodepng.h"
#include <sstream>
#include <vector>

using namespace cl;

// ======================================================
// ======================================================
void init_obstacle_mask(const LBMParams &params,
                        uint8_t *obstacle,
                        sycl::queue &q,
                        sycl::event &ev)
{

  const size_t nx = params.nx;
  const size_t ny = params.ny;

  const real_t cx = params.cx;
  const real_t cy = params.cy;

  const real_t r = params.r;

  ev = q.submit([&](sycl::handler &cgh)
                {
                  cgh.parallel_for(sycl::range<2>{ny, nx}, [=](sycl::id<2> id)
                                   {
                                     size_t i = id[1]; //along nx
                                     size_t j = id[0]; //along ny

                                     size_t index = i + nx * j;

                                     real_t x = 1.0 * i;
                                     real_t y = 1.0 * j;

                                     obstacle[index] = (x - cx) * (x - cx) + (y - cy) * (y - cy) < r * r ? 1 : 0;
                                   });
                });

} // init_obstacle_mask

// ======================================================
void initialize_macroscopic_variables(const LBMParams &params,
                                      real_t *rho,
                                      real_t *ux,
                                      real_t *uy,
                                      sycl::queue &q,
                                      std::vector<sycl::event> &evs)
{

  const size_t nx = params.nx;
  const size_t ny = params.ny;

  const double uLB = params.uLB;
  const double ly = params.ly;

  // rho is one everywhere : fill 1
  const real_t fillRho = 1.0;

  evs.emplace_back(q.submit([&](sycl::handler &cgh)
                            { cgh.fill<real_t>(rho, fillRho, nx * ny); }));

  // uy is zero everywhere : fill 0
  const real_t fillUy = 0.0;

  evs.emplace_back(q.submit([&](sycl::handler &cgh)
                            { cgh.fill<real_t>(uy, fillUy, nx * ny); }));

  // fill ux

  evs.emplace_back(q.submit([&](sycl::handler &cgh)
                            {
                              cgh.parallel_for(sycl::range<2>{ny, nx}, [=](sycl::id<2> id)
                                               {
                                                 size_t i = id[1]; //along nx
                                                 size_t j = id[0]; //along ny

                                                 size_t index = i + nx * j;

                                                 ux[index] = uLB * (1.0 + 1e-4 * sycl::sin((real_t)j / ly * 2 * M_PI));
                                               });
                            }));

} // initialize_macroscopic_variables

// ======================================================
void initialize_equilibrium(const LBMParams &params,
                            const real_t *v,
                            const real_t *t,
                            const real_t *rho,
                            const real_t *ux,
                            const real_t *uy,
                            real_t *fin,
                            sycl::queue &q,
                            sycl::event &ev)
{

  const size_t nx = params.nx;
  const size_t ny = params.ny;
  const int npop = LBMParams::npop;

  ev = q.submit([&](sycl::handler &cgh)
                {
                  cgh.parallel_for(sycl::range<2>{ny, nx}, [=](sycl::id<2> id)
                                   {
                                     size_t i = id[1]; //along nx
                                     size_t j = id[0]; //along ny

                                     size_t index = i + nx * j;

                                     real_t cu = 0.0;

                                     real_t uX = ux[index];
                                     real_t uY = uy[index];

                                     real_t usqr = 3.0 / 2 * (uX * uX + uY * uY);

                                     for (int ipop = 0; ipop < npop; ++ipop)
                                     {
                                       cu = 3 * (v[ipop * 2] * uX +
                                                 v[ipop * 2 + 1] * uY);
                                       // along the for loop, data is not contiguous. Data is contiguous along x, then nx-spaced
                                       // along y, and nx*ny spaced along npop dimension
                                       fin[index + ipop * nx * ny] = rho[index] * t[ipop] * (1 + cu + 0.5 * cu * cu - usqr);
                                     }
                                   });
                });
} // init equilibrium

// ==========================      LOOP        ============================

// ======================================================
void border_outflow(const LBMParams &params,
                    real_t *fin,
                    sycl::queue &q,
                    sycl::event &ev)
{

  const size_t nx = params.nx;
  const size_t ny = params.ny;

  const int nxny = nx * ny;

  ev = q.submit([&](sycl::handler &cgh)
                {
                  const int i1 = nx - 1;
                  const int i2 = nx - 2;

                  cgh.parallel_for(sycl::range<1>{ny}, [=](sycl::id<1> j)
                                   {
                                     int index1 = i1 + nx * j;
                                     int index2 = i2 + nx * j;

                                     fin[index1 + 6 * nxny] = fin[index2 + 6 * nxny];
                                     fin[index1 + 7 * nxny] = fin[index2 + 7 * nxny];
                                     fin[index1 + 8 * nxny] = fin[index2 + 8 * nxny];
                                   });
                });

} // border_outflow

// ======================================================
void macroscopic(const LBMParams &params,
                 const real_t *v,
                 const real_t *fin,
                 real_t *rho,
                 real_t *ux,
                 real_t *uy,
                 sycl::queue &q,
                 sycl::event &ev)
{

  const size_t nx = params.nx;
  const size_t ny = params.ny;
  const int npop = LBMParams::npop;

  ev = q.submit([&](sycl::handler &cgh)
                {
                  cgh.parallel_for(sycl::range<2>{ny, nx}, [=](sycl::id<2> id)
                                   {
                                     size_t i = id[1]; //along nx
                                     size_t j = id[0]; //along ny

                                     int index = i + nx * j;

                                     double rho_tmp = 0;
                                     double ux_tmp = 0;
                                     double uy_tmp = 0;
                                     double tempFin = 0.0;

                                     for (int ipop = 0; ipop < npop; ++ipop)
                                     {

                                       tempFin = fin[index + ipop * nx * ny];

                                       // Oth order moment
                                       rho_tmp += tempFin;

                                       // 1st order moment
                                       ux_tmp += v[ipop * 2] * tempFin;
                                       uy_tmp += v[ipop * 2 + 1] * tempFin;

                                     } // end for ipop

                                     rho[index] = rho_tmp;
                                     ux[index] = ux_tmp / rho_tmp;
                                     uy[index] = uy_tmp / rho_tmp;
                                   });
                });

} // init macroscopic

// ======================================================
void border_inflow(const LBMParams &params,
                   const real_t *fin,
                   real_t *rho,
                   real_t *ux,
                   real_t *uy,
                   sycl::queue &q,
                   sycl::event &ev)
{

  const size_t nx = params.nx;
  const size_t ny = params.ny;

  const double uLB = params.uLB;
  const double ly = params.ly;

  const int nxny = nx * ny;

  ev = q.submit([&](sycl::handler &cgh)
                {
                  cgh.parallel_for(sycl::range<1>{ny}, [=](sycl::id<1> j)
                                   {
                                     int index = nx * j;
                                     // Compute velocity (for ux only, since uy = 0 everywhere)
                                     ux[index] = uLB * (1.0 + 1e-4 * sycl::sin((real_t)j / ly * 2 * M_PI));

                                     rho[index] = 1 / (1 - ux[index]) *
                                                  (fin[index + 3 * nxny] + fin[index + 4 * nxny] + fin[index + 5 * nxny] +
                                                   2 * (fin[index + 6 * nxny] + fin[index + 7 * nxny] + fin[index + 8 * nxny]));
                                   });
                });

} // border_inflow

// ======================================================
void equilibrium(const LBMParams &params,
                 const real_t *v,
                 const real_t *t,
                 const real_t *rho,
                 const real_t *ux,
                 const real_t *uy,
                 real_t *feq,
                 sycl::queue &q,
                 sycl::event &ev)
{

  const size_t nx = params.nx;
  const size_t ny = params.ny;

  const int npop = LBMParams::npop;

  ev = q.submit([&](sycl::handler &cgh)
                {
                  cgh.parallel_for(sycl::range<2>{ny, nx}, [=](sycl::id<2> id)
                                   {
                                     size_t i = id[1]; //along nx
                                     size_t j = id[0]; //along ny

                                     int index = i + nx * j;

                                     real_t usqr = 3.0 / 2 * (ux[index] * ux[index] + uy[index] * uy[index]);
                                     for (int ipop = 0; ipop < npop; ++ipop)
                                     {
                                       real_t cu = 3 * (v[ipop * 2] * ux[index] +
                                                        v[ipop * 2 + 1] * uy[index]);

                                       feq[index + ipop * nx * ny] = rho[index] * t[ipop] * (1 + cu + 0.5 * cu * cu - usqr);
                                     }
                                   });
                });

} // equilibrium

// ======================================================
void update_fin_inflow(const LBMParams &params,
                       const real_t *feq,
                       real_t *fin,
                       sycl::queue &q,
                       sycl::event &ev)
{

  const size_t nx = params.nx;
  const size_t ny = params.ny;

  const int nxny = nx * ny;

  ev = q.submit([&](sycl::handler &cgh)
                {
                  cgh.parallel_for(sycl::range<1>{ny}, [=](sycl::id<1> j)
                                   {
                                     int index = nx * j;

                                     fin[index + 0 * nxny] = feq[index + 0 * nxny] + fin[index + 8 * nxny] - feq[index + 8 * nxny];
                                     fin[index + 1 * nxny] = feq[index + 1 * nxny] + fin[index + 7 * nxny] - feq[index + 7 * nxny];
                                     fin[index + 2 * nxny] = feq[index + 2 * nxny] + fin[index + 6 * nxny] - feq[index + 6 * nxny];
                                   });
                });

} // update_fin_inflow

// ======================================================
void compute_collision(const LBMParams &params,
                       const real_t *fin,
                       const real_t *feq,
                       real_t *fout,
                       sycl::queue &q,
                       sycl::event &ev)
{

  const size_t nx = params.nx;
  const size_t ny = params.ny;

  const int nxny = nx * ny;

  const int npop = LBMParams::npop;
  const double omega = params.omega;

  ev = q.submit([&](sycl::handler &cgh)
                {
                  cgh.parallel_for(sycl::range<2>{ny, nx}, [=](sycl::id<2> id)
                                   {
                                     size_t i = id[1]; //along nx
                                     size_t j = id[0]; //along ny

                                     int index = i + nx * j;
                                     int index_f = index - nxny;

                                     for (int ipop = 0; ipop < npop; ++ipop)
                                     {
                                       index_f += nxny;
                                       fout[index_f] = fin[index_f] - omega * (fin[index_f] - feq[index_f]);
                                     } // end for ipop
                                   });
                });

} // compute_collision

// ======================================================
void update_obstacle(const LBMParams &params,
                     const real_t *fin,
                     const uint8_t *obstacle,
                     real_t *fout,
                     sycl::queue &q,
                     sycl::event &ev)
{

  const size_t nx = params.nx;
  const size_t ny = params.ny;

  const int nxny = nx * ny;
  const int npop = LBMParams::npop;

  ev = q.submit([&](sycl::handler &cgh)
                {
                  cgh.parallel_for(sycl::range<2>{ny, nx}, [=](sycl::id<2> id)
                                   {
                                     size_t i = id[1]; //along nx
                                     size_t j = id[0]; //along ny

                                     int index = i + nx * j;

                                     if (obstacle[index] == 1)
                                     {
                                       for (int ipop = 0; ipop < npop; ++ipop)
                                       {

                                         int index_out = index + ipop * nxny;
                                         int index_in = index + (8 - ipop) * nxny;

                                         fout[index_out] = fin[index_in];

                                       } // end for ipop
                                     }
                                   });
                });

} // update_obstacle

// ======================================================
void streaming(const LBMParams &params,
               const real_t *v,
               const real_t *fout,
               real_t *fin,
               sycl::queue &q,
               sycl::event &ev)
{

  const size_t nx = params.nx;
  const size_t ny = params.ny;

  const int nxny = nx * ny;
  const int npop = LBMParams::npop;

  ev = q.submit([&](sycl::handler &cgh)
                {
                  cgh.parallel_for(sycl::range<2>{ny, nx}, [=](sycl::id<2> id)
                                   {
                                     size_t i = id[1]; //along nx
                                     size_t j = id[0]; //along ny

                                     int index = i + nx * j;

                                     int index_in, index_out, j_out, i_out = 0;

                                     for (int ipop = 0; ipop < npop; ++ipop)
                                     {
                                       index_in = index + ipop * nxny;
                                       i_out = i - v[2 * ipop];

                                       if (i_out < 0)
                                         i_out += nx;
                                       if (i_out > nx - 1)
                                         i_out -= nx;

                                       j_out = j - v[2 * ipop + 1];
                                       if (j_out < 0)
                                         j_out += ny;
                                       if (j_out > ny - 1)
                                         j_out -= ny;

                                       index_out = i_out + nx * j_out + ipop * nxny;

                                       fin[index_in] = fout[index_out];

                                     } // end for ipop
                                   });
                });

} // streaming

//--------------------------------------
// ================================    Output PNG (GPU)    =========================

void prepare_png_output(const LBMParams &params,
                        const real_t *ux,
                        const real_t *uy,
                        real_t *u2,
                        unsigned char *img,
                        real_t *max_value,
                        real_t *min_value,
                        sycl::queue &q,
                        std::vector<sycl::event> &evs)
{

  const size_t nx = params.nx;
  const size_t ny = params.ny;

  // Reduction max_value (max)

  evs.emplace_back(q.submit([&](sycl::handler &cgh)
                            {
                              cgh.parallel_for(sycl::nd_range<1>{ny * nx, 1},
                                               sycl::reduction(max_value, sycl::maximum<>()),
                                               [=](sycl::nd_item<1> item, auto &max_value)
                                               {
                                                 size_t index = item.get_global_id();

                                                 real_t uu2 = sycl::abs(ux[index]);

                                                 u2[index] = uu2;

                                                 max_value.combine(uu2);
                                               });
                            }));

  // Reduction min_value (min)

  evs.emplace_back(q.submit([&](sycl::handler &cgh)
                            {
                              cgh.parallel_for(sycl::nd_range<1>{ny * nx, 1},
                                               sycl::reduction(min_value, sycl::minimum<>()),
                                               [=](sycl::nd_item<1> item, auto &min_value)
                                               {
                                                 size_t index = item.get_global_id();

                                                 min_value.combine(u2[index]);
                                               });
                            }));

  evs.emplace_back(q.submit([&](sycl::handler &cgh)
                            {
                              cgh.parallel_for(sycl::range<2>{ny, nx}, [=](sycl::id<2> id)
                                               {
                                                 size_t i = id[1]; //along nx
                                                 size_t j = id[0]; //along ny

                                                 int index = i + nx * j;

                                                 // rescale velocity in 0-255 range
                                                 unsigned char value = static_cast<unsigned char>((u2[index] - *min_value) / (*max_value - *min_value) * 255);
                                                 img[0 + 4 * i + 4 * nx * j] = value;
                                                 img[1 + 4 * i + 4 * nx * j] = 255 - value;
                                                 img[2 + 4 * i + 4 * nx * j] = value;
                                                 img[3 + 4 * i + 4 * nx * j] = 255 - value;
                                               });
                            }));
  evs.at(2).wait({evs.at(0), evs.at(1)}); // Present kernel requires the 2 previous events to be complete

} // LBMSolver::output_png

void output_png_parallel(int iTime,
                         int nx,
                         int ny,
                         const unsigned char *img)
{
  std::cout << "Output data (PNG) at time " << iTime << "\n";

  // create png image buff
  std::vector<unsigned char> image;
  image.resize(nx * ny * 4);
  image.assign(img, img + (nx * ny * 4));

  std::ostringstream iTimeNum;
  iTimeNum.width(7);
  iTimeNum.fill('0');
  iTimeNum << iTime;

  std::string filename = "vel_gpu_" + iTimeNum.str() + ".png";

  // encode the image
  unsigned error = lodepng::encode(filename, image, nx, ny);

  //if there's an error, display it
  if (error)
    std::cout << "encoder error " << error << ": " << lodepng_error_text(error) << std::endl;
}

void computeElapsedTime(const sycl::event &ev, double &elapsedTime, bool increment)
{
  auto end = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
  auto start = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
  double duration = static_cast<double>((end - start) / 1.0e6);
  elapsedTime = increment ? (elapsedTime + duration) : duration;
}

void computeElapsedTime(std::vector<sycl::event> &evs, double &elapsedTime, bool increment)
{
  sycl::cl_ulong start, end;
  double duration{0.0};
  for (const sycl::event &ev : evs)
  {
    computeElapsedTime(ev, duration, true);
  }

  elapsedTime = increment ? (elapsedTime + duration) : duration;
}