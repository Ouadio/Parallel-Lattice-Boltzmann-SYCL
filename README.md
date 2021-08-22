# Parallel Lattice Boltzmann Method : a SYCL DPC++ Implementation (CPU + GPU)

A minimal POC for 2D LBM *(Lattice Boltzmann Method)* simulation parallelized using the cross-architecture Programming Model [DPC++](https://spec.oneapi.io/versions/latest/elements/dpcpp/source/index.html) *(by Intel)* following the [SYCL 2020](https://www.khronos.org/sycl/) *Khronos Standard*.

## Prepare and Build DPC++ Compiler  
Check/install first prerequisites and build DPC++ Toolchain following the [official documentation guide](https://intel.github.io/llvm-docs/GetStartedGuide.html#build-dpc-toolchain-with-libc-library).

## How to build with DPCPP as SYCL Compiler ?


Prepare build configuration ENV variables to use DPCPP clang compiler : 
```bash
export PATH=$DPCPP_HOME/llvm/build/bin:$PATH
export LD_LIBRARY_PATH=$DPCPP_HOME/llvm/build/lib:$LD_LIBRARY_PATH

export CXX=clang++
export CC=clang
```
Build project for GPU or CPU : 
```bash
mkdir build
cd build
cmake [-DUSE_GPU= OFF | ON] [-DENABLE_PROFILING = OFF | ON] [-DUSE_DOUBLE = OFF | ON]..
make
```
Run the simulation *(optionally pass .ini parameters meta data file)*

```
./src/runLBMSimulation [Path/to/parameters/file.ini]
```

