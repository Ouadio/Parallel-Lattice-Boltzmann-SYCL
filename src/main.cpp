#include "lbm/LBMSolver.h"
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <map>
#include <string>
#include "lbm/real_type.h"

using namespace std;
using namespace cl;

int main(int argc, char *argv[]) {
  string input_file = argc > 1 ? string(argv[1]) : "flowAroundCylinder.ini";

  ConfigMap configMap(input_file);

  // Create a LBMParams object
  LBMParams<real_t> params{};

  params.setup(configMap);

  params.print();

  // Instanciate solver class
  // Temporary hardcoded device selection (GPU)
  sycl::device device;
#ifdef SYCL_GPU
  device = sycl::gpu_selector{}.select_device();
#else
  device = sycl::cpu_selector{}.select_device();
#endif

  LBMSolver<real_t> mySolver{params, device};

  mySolver.printQueueInfo();

#ifndef PROFILE
  auto start = chrono::high_resolution_clock::now();

#endif
  // Run simulation
  mySolver.run();

#ifdef PROFILE
  // PRINT PROFILING RESULTS
  map<string, double>::iterator it = mySolver.elapsedTime.begin();
  double totalElapsedTime = 0.0;

  cout << "\n----------  Total elapsed time per routine  --------------"
       << endl;
  while (it != mySolver.elapsedTime.end()) {
    cout << it->first << " : " << it->second << endl;
    totalElapsedTime += it->second;
    it++;
  }

  cout << "\n----------  Average elapsed time per routine  --------------"
       << endl;

  it = mySolver.elapsedTime.begin();
  while (it != mySolver.elapsedTime.end()) {
    cout << "Avg " << it->first << " : "
         << COMPUTE_AVG_ELAPSED_TIME(it->first, mySolver.elapsedTime,
                                     mySolver.repetitionCount)
         << endl;
    it++;
  }
  cout << "\n--------------   Total elapsed simulation time   --------------"
       << endl;

  cout << "  " << totalElapsedTime << " ms" << endl;
#else
  auto end = chrono::high_resolution_clock::now();
  double elapsedTime =
      chrono::duration_cast<chrono::milliseconds>(end - start).count();
  cout << "\nTotal Elapsed time : \n   " << elapsedTime << " ms" << endl;
#endif

  // delete mySolver;

  return EXIT_SUCCESS;
}
