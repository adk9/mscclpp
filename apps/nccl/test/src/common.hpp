// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_NCCL_TESTS_COMMON_HPP_
#define MSCCLPP_NCCL_TESTS_COMMON_HPP_

#include <mscclpp/gpu.hpp>
#include <nccl.h>

#define MPICHECK(cmd)                                                  \
  do {                                                                 \
    int e = cmd;                                                       \
    if (e != MPI_SUCCESS) {                                            \
      printf("Failed: MPI error %s:%d '%d'\n", __FILE__, __LINE__, e); \
      exit(EXIT_FAILURE);                                              \
    }                                                                  \
  } while (0)

#define CUDACHECK(cmd)                                                                      \
  do {                                                                                      \
    cudaError_t e = cmd;                                                                    \
    if (e != cudaSuccess) {                                                                 \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
      exit(EXIT_FAILURE);                                                                   \
    }                                                                                       \
  } while (0)

#define NCCLCHECK(cmd)                                                                      \
  do {                                                                                      \
    ncclResult_t r = cmd;                                                                   \
    if (r != ncclSuccess) {                                                                 \
      printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r)); \
      exit(EXIT_FAILURE);                                                                   \
    }                                                                                       \
  } while (0)

namespace mscclpp_nccl_tests {

unsigned int randGetSeed();

void randSetSeed(unsigned int seed = 0);

int rand();

int randUniform(int min, int max);

class DefaultConnection {
 public:
  DefaultConnection(int argc, char* argv[]);
  ~DefaultConnection();

  int rank() const { return rank_; }

  int nRanks() const { return nRanks_; }

  int localRank() const { return localRank_; }

  int deviceId() const { return deviceId_; }

  ncclComm_t comm() const { return comm_; }

  void barrier() const;

  void log(const char* fmt, ...);

 private:
  int rank_;
  int nRanks_;
  int localRank_;
  int deviceId_;

  int sharedRand_;
  ncclUniqueId id_;
  ncclComm_t comm_;
};

class GpuArray {
 public:
  GpuArray(size_t bytes, size_t alignment = 256);
  ~GpuArray();

  void initRandomFloat(float min, float max);

 private:
  int deviceId_;
  size_t bytes_;
  void *rawPtr_;
  void *alignedPtr_;
};

}  // namespace mscclpp_nccl_tests

#endif  // MSCCLPP_NCCL_TESTS_COMMON_HPP_
