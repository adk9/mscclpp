// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "common.hpp"
#include <mpi.h>
#include <cstdlib>
#include <unistd.h>
#include <stdarg.h>
#include <cstdio>
#include <ctime>
#include <vector>

static uint64_t getHostHash(const char* string) {
  // Based on DJB2a, result = result * 33 ^ char
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++) {
    result = ((result << 5) + result) ^ string[c];
  }
  return result;
}

static void getHostName(char* hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i = 0; i < maxlen; i++) {
    if (hostname[i] == '.') {
      hostname[i] = '\0';
      return;
    }
  }
}

unsigned int gRandomSeed = 0;

namespace mscclpp_nccl_tests {

unsigned int randGetSeed() {
  return gRandomSeed;
}

void randSetSeed(unsigned int seed) {
  gRandomSeed = seed;
  ::srand(seed);
}

int rand() {
  return ::rand();
}

int randUniform(int min, int max) {
  return min + rand() % (max - min);
}

float randUniform(float min, float max) {
  return min + (max - min) * rand() / float(RAND_MAX);
}

DefaultConnection::DefaultConnection(int argc, char* argv[]) {
  // initializing MPI
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank_));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks_));

  // calculating localRank based on hostname which is used in selecting a GPU
  uint64_t hostHashs[nRanks_];
  char hostname[1024];
  getHostName(hostname, 1024);
  hostHashs[rank_] = getHostHash(hostname);
  MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
  localRank_ = 0;
  for (int p = 0; p < nRanks_; p++) {
    if (p == rank_) break;
    if (hostHashs[p] == hostHashs[rank_]) localRank_++;
  }
  deviceId_ = localRank_;

  CUDACHECK(cudaSetDevice(localRank_));

  // get NCCL unique ID at rank 0 and broadcast it to all others
  if (rank_ == 0) ncclGetUniqueId(&id_);
  MPICHECK(MPI_Bcast((void*)&id_, sizeof(id_), MPI_BYTE, 0, MPI_COMM_WORLD));

  if (rank_ == 0) sharedRand_ = std::time(nullptr);
  MPICHECK(MPI_Bcast((void*)&sharedRand_, sizeof(sharedRand_), MPI_BYTE, 0, MPI_COMM_WORLD));
  randSetSeed(sharedRand_ + localRank_);

  // initializing NCCL
  NCCLCHECK(ncclCommInitRank(&comm_, nRanks_, id_, rank_));
}

DefaultConnection::~DefaultConnection() {
  // finalizing NCCL
  ncclCommDestroy(comm_);

  // finalizing MPI
  MPICHECK(MPI_Finalize());
}

void DefaultConnection::barrier() const {
  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
}

void DefaultConnection::log(const char* fmt, ...) {
  // prepend rank to fmt
  std::vector<char> newFmt(strlen(fmt) + 16);
  snprintf(newFmt.data(), newFmt.size(), "[Rank %d] %s", rank_, fmt);

  va_list args;
  va_start(args, fmt);
  vprintf(newFmt.data(), args);
  va_end(args);
}

GpuArray::GpuArray(size_t bytes, size_t alignment) : bytes_(bytes) {
  CUDACHECK(cudaGetDevice(&deviceId_));
  CUDACHECK(cudaMalloc(&rawPtr_, bytes + alignment));
  alignedPtr_ = (void*)(((size_t)rawPtr_ + (alignment - 1)) / alignment * alignment);
}

GpuArray::~GpuArray() {
  CUDACHECK(cudaFree(rawPtr_));
}

void GpuArray::initRandomFloat(float min, float max) {
  std::vector<float> hostData(bytes_ / sizeof(float));
  for (size_t i = 0; i < hostData.size(); i++) {
    hostData[i] = randUniform(min, max);
  }
  CUDACHECK(cudaSetDevice(deviceId_));
  cudaStream_t s;
  CUDACHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  CUDACHECK(cudaMemcpyAsync(alignedPtr_, hostData.data(), bytes_, cudaMemcpyHostToDevice, s));
  CUDACHECK(cudaStreamSynchronize(s));
}

}  // namespace mscclpp_nccl_tests
