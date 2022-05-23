#include <math.h> // for M_PI = 3.1415....

#include "sycl_profiler_utils.h"
#include <iostream>
#include <sstream>
#include <vector>

void computeElapsedTime(const sycl::event &ev, double &elapsedTime,
                        bool increment) {
  auto end = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
  auto start =
      ev.get_profiling_info<sycl::info::event_profiling::command_start>();
  double duration = static_cast<double>((end - start) / 1.0e6);
  elapsedTime = increment ? (elapsedTime + duration) : duration;
}

void computeElapsedTime(std::vector<sycl::event> &evs, double &elapsedTime,
                        bool increment) {
  sycl::cl_ulong start, end;
  double duration{0.0};
  for (const sycl::event &ev : evs) {
    computeElapsedTime(ev, duration, true);
  }

  elapsedTime = increment ? (elapsedTime + duration) : duration;
}