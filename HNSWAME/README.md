This is the code of combination of HNSW and AME.


## Experimental Environment

All experiments are conducted on a server equipped with 2 Intel(R) Xeon(R) Silver 4210R CPUs, each of which has 10 cores, and 256 GB DRAM as the main memory. The OS version is CentOS 7.9.2009. All codes were written by {C++} and compiled by {g++ 11.4}. The SIMD instructions are enabled to accelerate the distance computations.

## Building Instruction

### Prerequisites

cmake g++ openmp mkl openblas gfortran pthread

### Compile

```
cd PPANNS/
cd build
cmake ..
make
```

## Usage

You can edit the file `run.sh` for your experiments.
```
./run.sh
```

Note : M, efc, efs are the parameters of HNSW, efsm is maximum efsearch, efss is the step length of efSearch.
s and beta are the parameters of DCPE.
k is the num of kANN.
ratio is the ratio of k' to k.
database, dataquery and groundtruth are the input files, please give the absolute paths.

And, you should edit the CMakeLists.txt to replace the path of libraries with your path.
