#include <cstdlib>
#include <string>
#include <iostream>
#include <chrono>
#include <ctime>
#include <map>
#include "lbm/LBMSolver.h"
#include <map>

using namespace std;

int main(int argc, char *argv[])
{
  std::string input_file = argc > 1 ? std::string(argv[1]) : "flowAroundCylinder.ini";

  ConfigMap configMap(input_file);

  // Create a LBMParams object
  LBMParams params = LBMParams();
  params.setup(configMap);

  params.print();

  // Instanciate solver class
  LBMSolver *mySolver = new LBMSolver(params);

  mySolver->printQueueInfo();

#ifndef PROFILE
  auto start = chrono::high_resolution_clock::now();

#endif
  // Run simulation
  mySolver->run();

#ifdef PROFILE
  // PRINT PROFILING RESULTS
  std::map<std::string, double>::iterator it = mySolver->elapsedTime.begin();
  double totalElapsedTime = 0.0;

  std::cout << "\n----------  Total elapsed time per routine  --------------" << std::endl;
  while (it != mySolver->elapsedTime.end())
  {
    std::cout << it->first << " : " << it->second << std::endl;
    totalElapsedTime += it->second;
    it++;
  }

  std::cout << "\n----------  Average elapsed time per routine  --------------" << std::endl;

  it = mySolver->elapsedTime.begin();
  while (it != mySolver->elapsedTime.end())
  {
    std::cout << "Avg " << it->first << " : " << COMPUTE_AVG_ELAPSED_TIME(it->first, mySolver->elapsedTime, mySolver->repetitionCount) << std::endl;
    it++;
  }
  std::cout << "\n--------------   Total elapsed simulation time   --------------" << std::endl;

  std::cout << "  " << totalElapsedTime << " ms" << std::endl;
#else
  auto end = chrono::high_resolution_clock::now();
  double elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << "\nTotal Elapsed time (std::chrono) : \n   " << elapsedTime << " ms" << std::endl;
#endif

  delete mySolver;

  return EXIT_SUCCESS;
}
