<p align="center">
  <img src="https://raw.githubusercontent.com/wudangt/cgrad/master/logo/logo.png" width = "30%" height = "30%">
</p>

# cgrad

#### A implementation of max_pooling op and an element-wise add op in C or CUDA C(add in the future)

###### This small computation graph which consists of a max_pooling op and an element-wise add op:
    `Formula: dst[32,64,56,56] = max_pooling(src1[32,64,112,112]) + src2[32,1,56,56]`

### Installation
#### Compiling With pure C
```bash
git clone https://github.com/wudangt/cgrad.git
cd cgrad
make
./cgrad

#### Compiling With openMP
change the first line of the makefile to use GPU compilation
vim Makefile
OPENMP=1
make
./cgrad

#### Compiling With CUDA
If you want to use openMP, you can change the first line of the makefile to compile with openMP
vim Makefile
GPU=1
make
./cgrad
```
### Output

```python
|>>>>>>------scr1 data with size=[32,64,112,112]--------<<<<<<|
|>>>>>>------scr2 data with size=[32,1,56,56]--------<<<<<<|
Op fusion on GPU Time elapsed 0.000044 sec
Check sum value is 2851602432.000000
Check sum of image validated
|>>>>>>------dst data with size=[32,64,56,56]--------<<<<<<|
Op fusion on CPU Time elapsed 0.140381 sec
The checksum after the max_pool is 720978944.000000


```

### Set the src1 or src2 batch, channels, height, width, padding, stride

```python
#define SRC1_BATCH      32
#define SRC1_CHANNELS   64
#define SRC1_HEIGHT	    112
#define SRC1_WIDTH	    112

#define SRC2_BATCH      32
#define SRC2_CHANNELS   1
#define SRC2_HEIGHT	    56
#define SRC2_WIDTH	    56


#define PADDING         1
#define STRIDE          2
#define POOL_SIZE      3

```

### Perfomance analysis

The max_pooling op and an element-wise add op are fused and implemented in two way：C and CUDA C, where the C version is accelerated by openMP. The runtimes of the different versions of the implementation are as follows (Please kindly note that all code is tested on Intel(R) Xeon(R) Gold 5115 CPU @ 2.40GHz and Tesla P100 PCIe 16GB, using input size (32,64,112,112), pooling size (3,3) for performance analysis):

| \ | C  | C (openMP)  |CUDA C  |
| :-----: | :-: | :-: |:-: |
| Seconds | 0.140381 sec| 0.037699 sec |0.000044 sec |

As you can see from the time dimension, the CUDA C version is more than eight thousand times faster than the C version with openMP, and C with openMP is more than three times faster than the pure C version.

Similarly, the performance of the C version of the program can be analysis via gprof：
```python
cgrad$ gprof cgrad gmon.out -p
Flat profile:
Each sample counts as 0.01 seconds.
  %   cumulative   self              self     total
 time   seconds   seconds    calls  Ts/call  Ts/call  name
 72.78      0.16     0.16                             forward_maxpool_plus_add_fusion_layer
 13.65      0.19     0.03                             validate_src_data
  9.10      0.21     0.02                             initial_src1
  4.55      0.22     0.01                             print_max_pool_plus_add_checksum
  0.00      0.22     0.00        1     0.00     0.00  forward_maxpool_plus_add_fusion_layer_kernel(int, int, int, int, int, int, int,                                double*, double*, double*)
  0.00      0.22     0.00        1     0.00     0.00  __device_stub__Z44forward_maxpool_plus_add_fusion_layer_kerneliiiiiiiPdS_S_(int                               , int, int, int, int, int, int, double*, double*, double*)
  0.00      0.22     0.00        1     0.00     0.00  __sti____cudaRegisterAll()
  0.00      0.22     0.00        1     0.00     0.00  cuda_gridsize

 %         the percentage of the total running time of the
time       program used by this function.
```
and the CUDA C version of the program can be analysis via nvprof：
```python
cgrad$ nvprof ./cgrad
==32982== NVPROF is profiling process 32982, command: ./cgrad
==32982== Profiling application: ./cgrad
==32982== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   61.75%  64.070ms         2  32.035ms  20.283ms  43.787ms  [CUDA memcpy HtoD]
                   37.46%  38.868ms         1  38.868ms  38.868ms  38.868ms  [CUDA memcpy DtoH]
                    0.80%  827.23us         1  827.23us  827.23us  827.23us  forward_maxpool_plus_add_fusion_layer_kernel(int, int, int, int, int, int, int, double*, double*, double*)
      API calls:   70.99%  263.41ms         3  87.803ms  175.99us  263.05ms  cudaMalloc
                   28.29%  104.96ms         3  34.987ms  20.551ms  43.963ms  cudaMemcpy
                    0.25%  925.48us         3  308.49us  224.31us  424.98us  cuDeviceTotalMem
                    0.23%  852.15us         2  426.07us  6.1070us  846.04us  cudaDeviceSynchronize
                    0.21%  772.21us       288  2.6810us     164ns  116.64us  cuDeviceGetAttribute
                    0.02%  82.685us         3  27.561us  22.842us  36.873us  cuDeviceGetName
                    0.01%  49.258us         1  49.258us  49.258us  49.258us  cudaLaunchKernel
                    0.00%  14.523us         3  4.8410us  1.4280us  10.497us  cuDeviceGetPCIBusId
                    0.00%  1.7160us         6     286ns     164ns     803ns  cuDeviceGet
                    0.00%  1.1850us         3     395ns     220ns     716ns  cuDeviceGetCount
                    0.00%     862ns         3     287ns     252ns     351ns  cuDeviceGetUuid
                    0.00%     375ns         1     375ns     375ns     375ns  cudaPeekAtLastError
                    0.00%     317ns         1     317ns     317ns     317ns  cudaGetLastError
```
From the output of the above analysis tool, it can be seen that the bottleneck of the pure C version of the code is that it cannot take advantage of the multi-threading of the cpu resulting in a large computational overhead of the operator, while the CUDA C version of the code can take full advantage of the multi-threading of the GPU with a smaller computational overhead, but at the same time the transfer of data between the host and the device takes up most of the time.
  ### Features
- [x] # Forward :tada:
- [x] # OpenMP :tada:
- [x] # op fusion :tada:
- [x] # Unit Test (need to add more unit test) :tada:
- [x] # cuda extension :tada:
- [ ] # backward
- [ ] # SIMD
- [ ] # cache locality  
- [ ] # memory management
### Future Work
- [ ] # How to use the data that is overlapped in the pooling process
- [ ] # Embedded as an operator in a Pytorch framework 
- [ ] # May be with Low precision BF16?
- [ ] # Shared memory
