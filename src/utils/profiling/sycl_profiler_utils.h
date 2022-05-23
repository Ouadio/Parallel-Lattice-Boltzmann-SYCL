#pragma once

#include "stdint.h"
#include <CL/sycl.hpp>

using namespace cl;

void computeElapsedTime(const sycl::event &ev, double &elapsedTime,
                        bool increment = false);
void computeElapsedTime(std::vector<sycl::event> &evs, double &elapsedTime,
                        bool increment = false);
