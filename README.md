<p align="center">
  <img src="https://raw.githubusercontent.com/wudangt/cgrad/master/logo/logo.png" width = "30%" height = "30%">
</p>

# cgrad

#### A implementation of max_pooling op and an element-wise add op in C or CUDA C(add in the future)

###### This small computation graph which consists of a max_pooling op and an element-wise add op:
    `Formula: dst[32,64,56,56] = max_pooling(src1[32,64,112,112]) + src2[32,1,56,56]`

### Installation
```bash
git clone https://github.com/wudangt/cgrad.git
cd cgrad
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
#define POOL_WIDTH      3
#define POOL_HEIGHT     3

```
### Perfomance analysis
The max_pooling op and an element-wise add op are fused and implemented in two way：C and CUDA C, where the C version is accelerated by openMP. The runtimes of the different versions of the implementation are as follows:

| \ | C  | C (openMP)  |CUDA C  |
| :-----: | :-: | :-: |:-: |
| Seconds | 0.140381 sec| 0.037699 sec |0.000044 sec |

As you can see from the time dimension, the CUDA C version is more than eight thousand times faster than the C version with openMP, and C with openMP is more than three times faster than the pure C version.
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
