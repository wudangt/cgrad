#ifndef CUDA_H
#define CUDA_H

#include "cgrad.h"

#ifdef GPU

void check_error(cudaError_t status);
dim3 cuda_gridsize(size_t n);


#ifdef CUDNN
cudnnHandle_t cudnn_handle();
#endif

#endif
#endif
