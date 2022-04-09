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
Check sum value is 2851602432.000000 
Check sum of image validated 
|>>>>>>------dst data with size=[32,64,56,56]--------<<<<<<|
Op fusion on CPU Time elapsed 0.081029 sec
The checksum after the max_pool is 725745664.000000 

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
