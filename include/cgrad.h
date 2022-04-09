#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <sys/time.h>

#ifdef GPU
    #define BLOCK 512
      #include <cuda_runtime.h>
      #include "curand.h"
      #include "cublas_v2.h"
    
#ifdef CUDNN
    #include "cudnn.h"
 #endif
#endif
void error(const char *s);
