// Code borrowed from https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html

#include "common.hpp"
#include <unistd.h>

namespace tests = mscclpp_nccl_tests;

int main(int argc, char* argv[]) {
  tests::DefaultConnection conn(argc, argv);

  conn.log("Seed = %u\n", tests::randGetSeed());


  int size = 32 * 1024 * 1024;

  float *sendbuff, *recvbuff;
  cudaStream_t s;

  // picking a GPU based on localRank, allocate device buffers
  CUDACHECK(cudaMalloc(&sendbuff, size * sizeof(float)));
  CUDACHECK(cudaMalloc(&recvbuff, size * sizeof(float)));
  CUDACHECK(cudaStreamCreate(&s));

  // communicating using NCCL
  NCCLCHECK(ncclAllReduce((const void*)sendbuff, (void*)recvbuff, size, ncclFloat, ncclSum, conn.comm(), s));

  // completing NCCL operation by synchronizing on the CUDA stream
  CUDACHECK(cudaStreamSynchronize(s));

  // free device buffers
  CUDACHECK(cudaFree(sendbuff));
  CUDACHECK(cudaFree(recvbuff));
  return 0;
}
