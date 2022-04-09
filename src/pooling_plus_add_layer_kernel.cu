#include <cuda_runtime.h>
#include "curand.h"
#include "cublas_v2.h"
extern "C" { 
	#include "pooling_plus_add_layer.h"
	#include "cuda.h"
}
__global__ void forward_maxpool_plus_add_fusion_layer_kernel(int n, int in_h, int in_w, int in_c, int stride, int size, int pad, double *src1_pointer, double *src2_pointer, double *dst_pointer)
{
    int h = (in_h + 2*pad - size)/stride + 1;
    int w = (in_w + 2*pad - size)/stride + 1;
    int c = in_c;

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
    int src2_index = j + w*(i + h*(k + b)); 
    double max = 0;
    int l, m;
    for(l = 0; l < size; ++l){
        for(m = 0; m < size; ++m){
            int cur_h = h_offset + i*stride + l;
            int cur_w = w_offset + j*stride + m;
            int index = cur_w + in_w*(cur_h + in_h*(k + b*in_c));
            int valid = (cur_h >= 0 && cur_h < in_h &&
                    cur_w >= 0 && cur_w < in_w);
            double val = (valid != 0) ? src1_pointer[index] : -INFINITY;
            max   = (val > max) ? val   : max;
        }
    }
    dst_pointer[src1_index] = max + src2_pointer[src2_index];
}


extern void forward_maxpool_plus_add_fusion_layer_gpu(int batch, int in_h, int in_w, int in_c, int stride, int size, int pad, double *src1_pointer, double *src2_pointer, double *dst_pointer)
{
   
    int out_h = (in_h + 2*pad - size)/stride + 1;
    
    int out_w = (in_w + 2*pad - size)/stride + 1;
    int out_c = in_c;

    size_t n = batch*out_h*out_w*out_c;
    forward_maxpool_plus_add_fusion_layer_kernel<<<cuda_gridsize(n), BLOCK>>>(n, in_h, in_w, in_c, stride, size, pad, src1_pointer, src2_pointer, dst_pointer);
    check_error(cudaPeekAtLastError());
}	
