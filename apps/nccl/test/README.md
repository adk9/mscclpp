# mscclpp-nccl-tests

## Build

* For NVIDIA:
    ```bash
    mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON ..
    make -j
    ```

* For AMD:
    ```bash
    mkdir build && cd build && CXX=/opt/rocm/bin/hipcc cmake -DCMAKE_BUILD_TYPE=Release -DUSE_ROCM=ON ..
    make -j
    ```

## Run test suites

Run `ctest` under the build directory.

To see details of tests, run in verbose mode.

```bash
ctest --verbose
```
