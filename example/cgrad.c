#include "cgrad.h"

#define CUDA_CALL(x) do {		 \
  cudaError_t ____rc = (x);		 \
  assert(____rc == cudaSuccess); \
} while (0)
/* src1 batch, channels, height, width, padding, stride */
#define SRC1_BATCH       32
#define SRC1_CHANNELS    64
#define SRC1_HEIGHT	    112
#define SRC1_WIDTH	    112

/* src1 batch, channels, height, width, padding, stride */
#define SRC2_BATCH       32
#define SRC2_CHANNELS    1
#define SRC2_HEIGHT	    56
#define SRC2_WIDTH	    56
/* Pool height, width, padding, stride */
#define PADDING     1
#define STRIDE      2
#define POOL_SIZE  3

extern double cpuSecond();
extern void initial_src(int batch, int channels, int height, int width, float *image_pointer, int value);
extern void validate_src_data(int batch, int channels, int height, int width, float *image_pointer);
extern void print_data(int batch, int channels,int height,int width, float *image_pointer);
extern void print_max_pool_plus_add_checksum(int batch, int channels, int height, int width, float *output_pointer);
extern void forward_maxpool_plus_add_fusion_layer_with_openmp(int batch, int src1_in_h, int src1_in_w, int src1_in_c, int src2_in_h, int src2_in_w, int src2_in_c, int stride, int size, int pad, float *src1_pointer, float *src2_pointer, float *dst_pointer);
extern void forward_maxpool_plus_add_fusion_layer_with_openmp_and_sse(int batch, int src1_in_h, int src1_in_w, int src1_in_c, int src2_in_h, int src2_in_w, int src2_in_c, int stride, int size, int pad, float *src1_pointer, float *src2_pointer, float *dst_pointer);
extern void forward_maxpool_plus_add_layer(int batch, int src1_in_h, int src1_in_w, int src1_in_c, int src2_in_h, int src2_in_w, int src2_in_c, int stride, int size, int pad, float *src1_pointer, float *src2_pointer, float *dst_pointer);
extern void forward_maxpool_plus_add_fusion_layer_gpu(int batch, int src1_in_h, int src1_in_w, int src1_in_c, int src2_in_h, int src2_in_w, int src2_in_c, int stride, int size, int pad, float *src1_pointer, float *src2_pointer, float *dst_pointer);
int main(int ac, char *av[]){
	static_assert(sizeof(int) >= 4, "Interger should have at least 4.");
	int pooled_height = (SRC1_HEIGHT + 2*PADDING - POOL_SIZE)/2 + 1;
        int pooled_width = (SRC1_WIDTH + 2*PADDING - POOL_SIZE)/2 + 1;
	int pooled_size = SRC1_BATCH*SRC1_CHANNELS*pooled_height*pooled_width*sizeof(float);	

	int src1_size = SRC1_BATCH*SRC1_CHANNELS*SRC1_HEIGHT*SRC1_WIDTH*sizeof(float);
        int src2_size = pooled_size;
	int dst_size = pooled_size;
	float *gpu_src1_pointer, *gpu_src2_pointer, *gpu_output_pointer;
	float *src1_pointer, *src2_pointer, *src1_output_pointer, *dst_pointer, *output_pointer;
	
	src1_pointer = (float *) malloc(src1_size);
	src1_output_pointer = (float *) malloc(pooled_size);
	src2_pointer = (float *) malloc(src2_size);
	dst_pointer = (float *) malloc(dst_size);
	output_pointer = (float *) malloc(src2_size);
	
	memset(src1_output_pointer, 0, src2_size);
	memset(dst_pointer, 0, dst_size);
	memset(output_pointer, 0, src2_size);
	
	initial_src(SRC1_BATCH, SRC1_CHANNELS, SRC1_HEIGHT, SRC1_WIDTH, src1_pointer, 5);
	initial_src(SRC2_BATCH, SRC2_CHANNELS, pooled_height, pooled_width, src2_pointer, 1);
	cudaMalloc((void **)&gpu_src1_pointer, src1_size);	
	cudaMalloc((void **)&gpu_src2_pointer, src2_size);
	cudaMalloc((void **)&gpu_output_pointer, src2_size);
	cudaMemcpy(gpu_src1_pointer, src1_pointer, src1_size, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_src2_pointer, src2_pointer, src2_size, cudaMemcpyHostToDevice);
	printf("|>>>>>>------scr1 data with size=[%d,%d,%d,%d]--------<<<<<<|\n",SRC1_BATCH, SRC1_CHANNELS, SRC1_HEIGHT, SRC1_WIDTH);
	//print_data(SRC1_BATCH, SRC1_CHANNELS, SRC1_HEIGHT, SRC1_WIDTH, src1_pointer);
	printf("|>>>>>>------scr2 data with size=[%d,%d,%d,%d]--------<<<<<<|\n",SRC2_BATCH, SRC2_CHANNELS, pooled_height, pooled_width);
	//print_data(SRC2_BATCH, SRC2_CHANNELS, pooled_height, pooled_width, src2_pointer);
	
	FILE *fpt;
	fpt = fopen("results/result18_24.txt","w+");
	fprintf(fpt,"id,cpu, openmp, openmp_sse, gpu \n");
	for(int i = 0;i < 10; i++ ){
	        double iStart_gpu = cpuSecond();
		forward_maxpool_plus_add_fusion_layer_gpu(SRC1_BATCH, SRC1_HEIGHT, SRC1_WIDTH, SRC1_CHANNELS, SRC2_HEIGHT, SRC2_WIDTH, SRC2_CHANNELS, STRIDE, POOL_SIZE, PADDING, gpu_src1_pointer, gpu_src2_pointer, gpu_output_pointer);
		double iElaps_gpu = cpuSecond()-iStart_gpu;	
		printf("Op fusion on GPU Time elapsed %f sec\n", iElaps_gpu);
		cudaDeviceSynchronize();
		cudaMemcpy(output_pointer, gpu_output_pointer, src2_size, cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		double iStart_cpu_openmp = cpuSecond();
		forward_maxpool_plus_add_fusion_layer_with_openmp(SRC1_BATCH, SRC1_HEIGHT,SRC1_WIDTH, SRC1_CHANNELS, SRC2_HEIGHT, SRC2_WIDTH, SRC2_CHANNELS, STRIDE, POOL_SIZE, PADDING, src1_pointer, src2_pointer, dst_pointer);
		double iElaps_cpu_openmp = cpuSecond()-iStart_cpu_openmp;
	

		double iStart_cpu_openmp_sse = cpuSecond();
		forward_maxpool_plus_add_fusion_layer_with_openmp_and_sse(SRC1_BATCH, SRC1_HEIGHT,SRC1_WIDTH, SRC1_CHANNELS, SRC2_HEIGHT, SRC2_WIDTH, SRC2_CHANNELS, STRIDE, POOL_SIZE, PADDING, src1_pointer, src2_pointer, dst_pointer);
		double iElaps_cpu_openmp_sse = cpuSecond()-iStart_cpu_openmp_sse;
		

	
		double iStart_cpu_c = cpuSecond();
		forward_maxpool_plus_add_layer(SRC1_BATCH, SRC1_HEIGHT,SRC1_WIDTH, SRC1_CHANNELS, SRC2_HEIGHT, SRC2_WIDTH, SRC2_CHANNELS, STRIDE, POOL_SIZE, PADDING, src1_pointer, src2_pointer, dst_pointer);
		double iElaps_cpu_c = cpuSecond()-iStart_cpu_c;
		fprintf(fpt, "%d %lf, %lf, %lf, %lf\n",i, iElaps_cpu_c, iElaps_cpu_openmp,iElaps_cpu_openmp_sse,iElaps_gpu);	
		//validate_src_data(SRC1_BATCH, SRC1_CHANNELS, SRC1_HEIGHT, SRC1_WIDTH, src1_pointer);
		//printf("|>>>>>>------dst data with size=[%d,%d,%d,%d]--------<<<<<<|\n",SRC1_BATCH, SRC1_CHANNELS, pooled_height, pooled_width);
		//print_data(SRC1_BATCH, SRC1_CHANNELS, pooled_height, pooled_width, dst_pointer);
		//print_data(SRC1_BATCH, SRC1_CHANNELS, pooled_height, pooled_width, output_pointer);
		printf("Op  on CPU Time elapsed %f sec\n", iElaps_cpu_c);
		printf("Op fusion on CPU Time elapsed %f sec\n", iElaps_cpu_openmp);
        	//print_max_pool_plus_add_checksum(SRC1_BATCH, SRC1_CHANNELS, SRC2_HEIGHT, SRC2_WIDTH, dst_pointer);
        	//print_max_pool_plus_add_checksum(SRC1_BATCH, SRC1_CHANNELS, SRC2_HEIGHT, SRC2_WIDTH, output_pointer);
	}
	fclose(fpt);
	free(src1_pointer);
        free(src2_pointer);
        free(src1_output_pointer);
        free(dst_pointer);
  	
}

