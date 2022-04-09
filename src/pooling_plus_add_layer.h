#ifndef MAXPOOL_PLUS_ADD_LAYER_H
#define MAXPOOL_PLUS_ADD_LAYER_H
#include <omp.h>
void forward_maxpool_plus_add_fusion_layer(int batch, int in_h, int in_w, int in_c, int stride, int size, int pad, double *src1_pointer, double *src2_pointer, double *dst_pointer);

#ifdef GPU
//void backward_maxpool_layer_gpu(maxpool_layer l, network net);
void forward_maxpool_plus_add_fusion_layer_gpu(int batch, int in_h, int in_w, int in_c, int stride, int size, int pad, double *src1_pointer, double *src2_pointer, double *dst_pointer);
#endif



#endif
