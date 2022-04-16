#include <cuda_runtime.h>
#include "curand.h"
#include "cublas_v2.h"
extern "C" { 
	#include "pooling_plus_add_layer.h"
	#include "cuda.h"
}
__global__ void forward_maxpool_plus_add_fusion_layer_kernel(int n, int src1_in_h, int src1_in_w, int src1_in_c, int src2_in_h, int src2_in_w, int src2_in_c, int stride, int size, int pad, float *src1_pointer, float *src2_pointer, float *dst_pointer)
{
    int h = (src1_in_h + 2*pad - size)/stride + 1;
    int w = (src1_in_w + 2*pad - size)/stride + 1;
    int c = src1_in_c;

    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;

    int j = id % w;
    id /= w;
    int i = id % h;
    id /= h;
    int k = id % c;
    id /= c;
    int b = id;

    int w_offset = -pad;
    int h_offset = -pad;

    int src1_index = j + w*(i + h*(k + c*b));
    int src2_index = (j + src2_in_w)%src2_in_w + src2_in_w*( (i + src2_in_h)%src2_in_h + src2_in_h*( (k + src2_in_c)%src2_in_c + src2_in_c*b));
    float max = 0;
    int l, m;
    for(l = 0; l < size; ++l){
        for(m = 0; m < size; ++m){
            int cur_h = h_offset + i*stride + l;
            int cur_w = w_offset + j*stride + m;
            int index = cur_w + src1_in_w*(cur_h + src1_in_h*(k + b*src1_in_c));
            int valid = (cur_h >= 0 && cur_h < src1_in_h &&
                    cur_w >= 0 && cur_w < src1_in_w);
            float val = (valid != 0) ? src1_pointer[index] : -INFINITY;
            max   = (val > max) ? val   : max;
        }
    }
    dst_pointer[src1_index] = max + src2_pointer[src2_index];
}


extern  void forward_maxpool_plus_add_fusion_layer_gpu(int batch, int src1_in_h, int src1_in_w, int src1_in_c, int src2_in_h, int src2_in_w, int src2_in_c, int stride, int size, int pad, float *src1_pointer, float *src2_pointer, float *dst_pointer)
{   
    int out_h = (src1_in_h + 2*pad - size)/stride + 1;
    
    int out_w = (src1_in_w + 2*pad - size)/stride + 1;
    int out_c = src1_in_c;

    size_t n = batch*out_h*out_w*out_c;
    forward_maxpool_plus_add_fusion_layer_kernel<<<cuda_gridsize(n), BLOCK>>>(n, src1_in_h, src1_in_w, src1_in_c, src2_in_h, src2_in_w, src2_in_c, stride, size, pad, src1_pointer, src2_pointer, dst_pointer);
    check_error(cudaPeekAtLastError());
}	
