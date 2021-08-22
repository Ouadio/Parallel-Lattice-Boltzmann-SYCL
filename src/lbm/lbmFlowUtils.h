#ifndef LBM_FLOW_UTILS_H
#define LBM_FLOW_UTILS_H

#include "LBMParams.h"
#include "stdint.h"
#include <CL/sycl.hpp>

using namespace cl;

/**
 * Compute equilibrium disbution function for a given set of macroscopic
 * variables.
 *
 * \param[in] params LBM parameters
 * \param[in] v lattice velocities
 * \param[in] rho macroscopic density
 * \param[in] ux X-component of macroscopic velocity
 * \param[in] uy Y-component of macroscopic velocity
 * \param[out] feq distribution functions
 */
void equilibrium(const LBMParams &params,
                 const real_t *v,
                 const real_t *t,
                 const real_t *rho,
                 const real_t *ux,
                 const real_t *uy,
                 real_t *feq,
                 sycl::queue &q,
                 sycl::event &ev);

/**
 * Compute initial equilibrium disbution function for the initial set of macroscopic
 * variables.
 *
 * \param[in] params LBM parameters
 * \param[in] v lattice velocities
 * \param[in] rho macroscopic density
 * \param[in] ux X-component of macroscopic velocity
 * \param[in] uy Y-component of macroscopic velocity
 * \param[out] feq distribution functions
 */
void initialize_equilibrium(const LBMParams &params,
                            const real_t *v,
                            const real_t *t,
                            const real_t *rho,
                            const real_t *ux,
                            const real_t *uy,
                            real_t *fin,
                            sycl::queue &q,
                            sycl::event &ev);
/**
 * Setup: cylindrical obstacle mask.
 *
 * \param[in] params
 * \param[out] obstacle
 */
void init_obstacle_mask(const LBMParams &params,
                        uint8_t *obstacle,
                        sycl::queue &q,
                        sycl::event &ev);

/**
 * initialize macroscopic variables.
 */
void initialize_macroscopic_variables(const LBMParams &params,
                                      real_t *rho,
                                      real_t *ux,
                                      real_t *uy,
                                      sycl::queue &q,
                                      std::vector<sycl::event> &evs);

/**
 * border condition : outflow on the right interface
 *
 * Right wall: outflow condition.
 * we only need here to specify distrib. function for velocities
  that enter the domain (other that go out, are set by the streaming step)
 */
void border_outflow(const LBMParams &params,
                    real_t *fin,
                    sycl::queue &q,
                    sycl::event &ev);

/**
 * Compute macroscopic variables from distribution functions.
 *
 * fluid density is 0th moment of distribution functions
 * fluid velocity components are 1st order moment of dist. functions
 *
 * \param[in] params LBM parameters
 * \param[in] v lattice velocities
 * \param[in] fin distribution functions
 * \param[out] rho macroscopic density
 * \param[out] ux X-component of macroscopic velocity
 * \param[out] uy Y-component of macroscopic velocity
 */
void macroscopic(const LBMParams &params,
                 const real_t *v,
                 const real_t *fin,
                 real_t *rho,
                 real_t *ux,
                 real_t *uy,
                 sycl::queue &q,
                 sycl::event &ev);

/**
 * border condition : inflow on the left interface
 */
void border_inflow(const LBMParams &params,
                   const real_t *fin,
                   real_t *rho,
                   real_t *ux,
                   real_t *uy,
                   sycl::queue &q,
                   sycl::event &ev);

/**
 * Update fin at inflow left border.
 */
void update_fin_inflow(const LBMParams &params,
                       const real_t *feq,
                       real_t *fin,
                       sycl::queue &q,
                       sycl::event &ev);

/**
 * Compute collision
 */
void compute_collision(const LBMParams &params,
                       const real_t *fin,
                       const real_t *feq,
                       real_t *fout,
                       sycl::queue &q,
                       sycl::event &ev);

/**
 * Update distrib. function inside obstacle.
 */
void update_obstacle(const LBMParams &params,
                     const real_t *fin,
                     const uint8_t *obstacle,
                     real_t *fout,
                     sycl::queue &q,
                     sycl::event &ev);

/**
 * Streaming.
 */
void streaming(const LBMParams &params,
               const real_t *v,
               const real_t *fout,
               real_t *fin,
               sycl::queue &q,
               sycl::event &ev);

/**
 * PNG Output in Parallel fashion.
 */
void prepare_png_output(const LBMParams &params,
                        const real_t *ux,
                        const real_t *uy,
                        real_t *u2,
                        unsigned char *img,
                        real_t *max_value,
                        real_t *min_value,
                        sycl::queue &q,
                        std::vector<sycl::event> &evs);

void output_png_parallel(int iTime,
                         int nx,
                         int ny,
                         const unsigned char *img);

void computeElapsedTime(const sycl::event &ev, double &elapsedTime, bool increment = false);
void computeElapsedTime(std::vector<sycl::event> &evs, double &elapsedTime, bool increment = false);

#endif // LBM_FLOW_UTILS_H
