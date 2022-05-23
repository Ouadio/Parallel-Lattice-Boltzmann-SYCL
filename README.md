# Parallel Lattice Boltzmann Method : a SYCL Portable Implementation 

A simple 2D Fluid Dynamics Simulation using the LBM *(Lattice Boltzmann Method)*, implemented with the SYCL heterogeneous programming model, and able to run on multiple devices including intel's CPUs and GPUs and Nvidia GPUs.


## 1. Build using ComputeCpp (Codeplay Software)
Download and install ComputeCpp CE Edition along with its requirements *(OpenCL Headers etc..)* following the [official documentation guide.](https://developer.codeplay.com/products/computecpp/ce/guides/#step-1-download-computecpp-community-edition)


```bash
export PATH=/path/to/computecpp/bin:$PATH
export LD_LIBRARY_PATH=/path/to/computecpp/lib:$LD_LIBRARY_PATH
export COMPUTECPP_DIR=/path/to/computecpp
export CXX=compute++
export CC=gcc

```
Build project for intel's GPU or CPU : 

```bash
mkdir build && cd build
cmake ../ -DUSE_COMPUTECPP=ON [-DUSE_GPU= OFF | ON] [-DENABLE_PROFILING = OFF | ON] [-DUSE_DOUBLE = OFF | ON].. && make

```
## 2. Build Using DPCPP (intel)   
Check/install first prerequisites and build DPC++ Toolchain following the [official documentation guide](https://intel.github.io/llvm-docs/GetStartedGuide.html#build-dpc-toolchain-with-libc-library).

## How to build with DPCPP/OneAPI ?

Prepare build configuration ENV variables to use DPCPP clang compiler : 
```bash
export PATH=$DPCPP_HOME/llvm/build/bin:$PATH
export LD_LIBRARY_PATH=$DPCPP_HOME/llvm/build/lib:$LD_LIBRARY_PATH

export CXX=clang++
export CC=clang
```
Build project for intel's GPU or CPU : 
```bash
mkdir build && cd build
cmake ../ [-DUSE_GPU= OFF | ON] [-DENABLE_PROFILING = OFF | ON] [-DUSE_DOUBLE = OFF | ON].. && make
```

Run the simulation *(optionally pass .ini parameters meta data file)*

```
./src/runLBMSimulation [Path/to/parameters/file.ini]
```

