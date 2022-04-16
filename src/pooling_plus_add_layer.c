#include "cuda.h"
#include "pooling_plus_add_layer.h"
#include <float.h>

void forward_maxpool_plus_add_fusion_layer(int batch, int src1_in_h, int src1_in_w, int src1_in_c, int src2_in_h, int src2_in_w, int src2_in_c, int stride, int size, int pad, float *src1_pointer, float *src2_pointer, float *dst_pointer)
{
    int b,i,j,k,m,n;
    int h = (src1_in_h + 2*pad - size)/stride + 1;
    int w = (src1_in_w + 2*pad - size)/stride + 1;
    int w_offset = -pad;
    int h_offset = -pad;
    #pragma omp parallel for
    for(b = 0; b < batch; ++b){
        for(k = 0; k < src1_in_c; ++k){
            for(i = 0; i < h; ++i){
                for(j = 0; j < w; ++j){
                    int src1_index = j + w*(i + h*(k + src1_in_c*b));  
                    int src2_index = (j + src2_in_w)%src2_in_w + src2_in_w*( (i + src2_in_h)%src2_in_h + src2_in_h*( (k + src2_in_c)%src2_in_c + src2_in_c*b));
		    float max = -FLT_MAX;
		    int s_start  = (i == 0 || j == 0) ? 0 : (size - stride);
                    #pragma unroll
		    for(n = s_start; n < size; ++n){
			#pragma unroll
                        for(m = s_start; m < size; ++m){
                            int cur_h = h_offset + i*stride + n;
                            int cur_w = w_offset + j*stride + m;
                            int index = cur_w + src1_in_w*(cur_h + src1_in_h*(k + b*src1_in_c));
                            int valid = (cur_h >= 0 && cur_h < src1_in_h &&
                                         cur_w >= 0 && cur_w < src1_in_w);
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
