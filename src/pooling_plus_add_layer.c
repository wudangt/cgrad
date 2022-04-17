#include "cuda.h"
#include "pooling_plus_add_layer.h"
#include <float.h>
#include <pmmintrin.h>

void forward_maxpool_plus_add_fusion_layer_with_openmp(int batch, int src1_in_h, int src1_in_w, int src1_in_c, int src2_in_h, int src2_in_w, int src2_in_c, int stride, int size, int pad, float *src1_pointer, float *src2_pointer, float *dst_pointer)
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

void forward_maxpool_plus_add_fusion_layer_with_openmp_and_sse(int batch, int src1_in_h, int src1_in_w, int src1_in_c, int src2_in_h, int src2_in_w, int src2_in_c, int stride, int size, int pad, float *src1_pointer, float *src2_pointer, float *dst_pointer)
{  
    assert(NULL = src1_pointer); 
    assert(NULL = src2_pointer); 
    assert(NULL = dst_pointer);
    assert(batch>0);
    assert(src1_in_h > src2_in_h);
    assert(src1_in_w > src2_in_w);
    assert(src1_in_c > src2_in_c);
    assert(stride > 0);
    assert(size >= 2);
    assert(pad > 0);
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
		    __m128 max = _mm_set1_ps(-FLT_MAX);
		    int s_start  = (i == 0 || j == 0) ? 0 : (size - stride);
                    #pragma unroll
		    for(n = s_start; n < size; n+=2){
			//#pragma unroll
                        for(m = s_start; m < size; m+=2){
                            int cur_h = h_offset + i*stride + n;
                            int cur_w = w_offset + j*stride + m;
                            int cur_w_r = w_offset + j*stride + m + 1;
                            int cur_h_d = h_offset + i*stride + n + 1;
			    
                            int index = cur_w + src1_in_w*(cur_h + src1_in_h*(k + b*src1_in_c));
			    int index_r = cur_w_r + src1_in_w*(cur_h + src1_in_h*(k + b*src1_in_c));
                            int index_d = cur_h + src1_in_w*(cur_h_d + src1_in_h*(k + b*src1_in_c));
			    int valid = (cur_h >= 0 && cur_h < src1_in_h &&
                                         cur_w >= 0 && cur_w < src1_in_w &&cur_w_r>=0 && cur_w_r<src1_in_w && cur_h_d>=0 && cur_h_d<src1_in_h);
			    __m128 vdata_1 = _mm_loadu_ps((float*) &src1_pointer[index]);
			    __m128 vdata_2 = _mm_loadu_ps((float*) &src1_pointer[index_r]);
			    __m128 vdata_3 = _mm_loadu_ps((float*) &src1_pointer[index_d]);
			    __m128 max_temp = _mm_max_ps(vdata_1, vdata_2);
			    max_temp = _mm_max_ps(max_temp, vdata_3);
                            __m128 val = (valid != 0) ?  max_temp : _mm_set1_ps(-FLT_MAX);
			     max = _mm_max_ps(val, max);
			    
                        }
                    }
		    __m128 vdata_4 = _mm_loadu_ps((float*) &src2_pointer[src2_index]);
		    __m128 vres = _mm_add_ps(max, vdata_4);
                    _mm_store_ss((float*) &dst_pointer[src1_index], vres);
                }
            }
        }
    }
    
}




void forward_maxpool_plus_add_layer(int batch, int src1_in_h, int src1_in_w, int src1_in_c, int src2_in_h, int src2_in_w, int src2_in_c, int stride, int size, int pad, float *src1_pointer, float *src2_pointer, float *dst_pointer)
{
    int b,i,j,k,m,n;
    int h = (src1_in_h + 2*pad - size)/stride + 1;
    int w = (src1_in_w + 2*pad - size)/stride + 1;
    int w_offset = -pad;
    int h_offset = -pad;
    for(b = 0; b < batch; ++b){
        for(k = 0; k < src1_in_c; ++k){
            for(i = 0; i < h; ++i){
                for(j = 0; j < w; ++j){
                    int src1_index = j + w*(i + h*(k + src1_in_c*b));  
		    float max = -FLT_MAX;
		    int s_start  = (i == 0 || j == 0) ? 0 : (size - stride);
		    for(n = s_start; n < size; ++n){
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
                    dst_pointer[src1_index] = max;
                }
            }
        }
    }

    for(b = 0; b < batch; ++b){
        for(k = 0; k < src1_in_c; ++k){
            for(i = 0; i < h; ++i){
                for(j = 0; j < w; ++j){
                    int src1_index = j + w*(i + h*(k + src1_in_c*b));  
                    int src2_index = (j + src2_in_w)%src2_in_w + src2_in_w*( (i + src2_in_h)%src2_in_h + src2_in_h*( (k + src2_in_c)%src2_in_c + src2_in_c*b));
                    dst_pointer[src1_index] += src2_pointer[src2_index];
                }
            }
        }
    }
    
}
