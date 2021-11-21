# simpleMPI with zero-copy

## Description

Simple example demonstrating how to use MPI in combination with CUDA.

This code is modified from NVIDIA's simpleMPI example to for my Jetson cluster's evaluation with zero-copy.
But this code operation is not limited to the jetson only, so I determined to remain most of NVIDIA's codes.

In general, mpi awares NVIDIA GPU (CUDA). When the hardware supports GPU Direct, GPU RDMA provides efficient data transfer between the GPUs. However, Jetson's networking does not provide such environment. Instead, we can utilize integrated memory architecture to minimize data transfer between the CPU and the GPU.

## Key Concepts

CUDA Systems Integration, MPI, Multithreading, Zero-copy

## Supported SM Architectures

[SM 3.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 3.7 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.1 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.6 ](https://developer.nvidia.com/cuda-gpus)

### When you are using Jetson (like my use case)

SM 7.2: Jetson Xavier NX, Jetson AGX Xavier
SM 6.2: Jetson TX2
SM 5.2: Jetson Nano, Jetson TX1

## Supported OSes

Linux 

## Supported CPU Architecture

x86_64, ppc64le, armv7l

## CUDA APIs involved

### [CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)
cudaMallco, cudaFree, cudaMemcpy

## Dependencies needed to build/run
[MPI](../../README.md#mpi)

## Prerequisites

Download and install the [CUDA Toolkit 11.5](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.
Make sure the dependencies mentioned in [Dependencies]() section above are installed.

## Build and Run

### Linux
The Linux samples are built using makefiles. To use the makefiles, change the current directory to the sample directory you wish to build, and run make:
```
$ cd <sample_dir>
$ make
```
The samples makefiles can take advantage of certain options:
*  **TARGET_ARCH=<arch>** - cross-compile targeting a specific architecture. Allowed architectures are x86_64, ppc64le, armv7l.
    By default, TARGET_ARCH is set to HOST_ARCH. On a x86_64 machine, not setting TARGET_ARCH is the equivalent of setting TARGET_ARCH=x86_64.<br/>
`$ make TARGET_ARCH=x86_64` <br/> `$ make TARGET_ARCH=ppc64le` <br/> `$ make TARGET_ARCH=armv7l` <br/>
    See [here](http://docs.nvidia.com/cuda/cuda-samples/index.html#cross-samples) for more details.
*   **dbg=1** - build with debug symbols
    ```
    $ make dbg=1
    ```
*   **SMS="A B ..."** - override the SM architectures for which the sample will be built, where `"A B ..."` is a space-delimited list of SM architectures. For example, to generate SASS for SM 50 and SM 60, use `SMS="50 60"`.
    ```
    $ make SMS="50 60"
    ```

  - If your target is Jetson devices, you should provide SM architectures as follow,
    ```
    $ make SMS="72 62 52"
    ```
    Because, current version of [Jetpack SDK](https://developer.nvidia.com/embedded/jetpack) 4.6 provides CUDA 10.2 and it does not support SM 8.0+ CUDA architectures.


*  **HOST_COMPILER=<host_compiler>** - override the default g++ host compiler. See the [Linux Installation Guide](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements) for a list of supported host compilers.
```
    $ make HOST_COMPILER=g++
```

## References (for more details)

