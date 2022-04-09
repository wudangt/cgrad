#include "cuda.h"
#include "pooling_plus_add_layer.h"
#include <float.h>


void forward_maxpool_plus_add_fusion_layer(int batch, int in_h, int in_w, int in_c, int stride, int size, int pad, double *src1_pointer, double *src2_pointer, double *dst_pointer)
{
    int b,i,j,k,m,n;
    int h = (in_h + 2*pad - size)/stride + 1;
    int w = (in_w + 2*pad - size)/stride + 1;

    int w_offset = -pad;
    int h_offset = -pad;
   #pragma omp parallel for
    for(b = 0; b < batch; ++b){
        for(k = 0; k < in_c; ++k){
            for(i = 0; i < h; ++i){
                for(j = 0; j < w; ++j){
                    int src1_index = j + w*(i + h*(k + in_c*b));  
                    int src2_index = j + w*(i + h*(k + b));
		    float max = -FLT_MAX;
		    #pragma unroll
                    for(n = 0; n < size; ++n){
			#pragma unroll
                        for(m = 0; m < size; ++m){
                            int cur_h = h_offset + i*stride + n;
                            int cur_w = w_offset + j*stride + m;
                            int index = cur_w + in_w*(cur_h + in_h*(k + b*in_c));
                            int valid = (cur_h >= 0 && cur_h < in_h &&
                                         cur_w >= 0 && cur_w < in_w);
                            float val = (valid != 0) ? src1_pointer[index] : -FLT_MAX;
                            max   = (val > max) ? val   : max;
                        }
                    }
                    dst_pointer[src1_index] = max + src2_pointer[src2_index];
                }
            }
        }
    }
}
