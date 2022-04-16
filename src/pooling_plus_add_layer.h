#ifndef MAXPOOL_PLUS_ADD_LAYER_H
#define MAXPOOL_PLUS_ADD_LAYER_H
#include <omp.h>
void forward_maxpool_plus_add_fusion_layer(int batch, int in_h, int in_w, int in_c, int stride, int size, int pad, float *src1_pointer, float *src2_pointer, float *dst_pointer);

#ifdef GPU
void forward_maxpool_plus_add_fusion_layer_gpu(int batch, int in_h, int in_w, int in_c, int stride, int size, int pad, float *src1_pointer, float *src2_pointer, float *dst_pointer);
#endif



#endif
