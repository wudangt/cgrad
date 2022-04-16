#include "cgrad.h"
#include "utils.h"

void error(const char *s)
{
    perror(s);
    assert(0);
    exit(-1);
}

double cpuSecond(){
	struct timeval tp;
	gettimeofday(&tp,NULL);
	return ((double)tp.tv_sec +(double)tp.tv_usec*1.e-6);
}

void initial_src1(int batch, int channels, int height, int width, float *image_pointer)
{
  int image_memory_size = batch*channels*height*width*sizeof(float);
  memset(image_pointer, 0, image_memory_size);
  	for(int b = 0; b<batch;b++){	
	  for(int k = 0; k < channels; k++){
	  	for(int i = 0; i < height; i++){
	  		for(int j = 0; j < width; j++){
	  			int index = b*channels*height*width + i*width + j + k*height*width;
	  			image_pointer[index] = (i+j);
	  		}
	  	}
	  }
	}
}

void initial_src2(int batch, int channels, int height, int width, float *image_pointer)
{
	int image_memory_size = batch*channels*height*width*sizeof(float);
	memset(image_pointer, 0, image_memory_size);
	for(int b = 0; b<batch; b++){	
	  for(int k = 0; k < channels; k++){
	  	for(int i = 0; i < height; i++){
	  		for(int j = 0; j < width; j++){
	  			int index = b*channels*height*width + i*width + j + k*height*width;
	  			image_pointer[index] = 1.0;
	  		}
	  	}
	  }
	}
}

void validate_src_data(int batch, int channels, int height, int width, float *image_pointer){
	float sum = 0.0;
	for(int b =0; b<batch;b++){
		for(int k = 0; k < channels; k++)
			for(int i = 0; i < height; i++){
	  			for(int j = 0; j < width; j++){
	  			{
	  				int index = b*channels*height*width + i*width + j + k*width*height;
	  				sum = sum + image_pointer[index];
	  			}
	  		}
	  	}
	}
  	printf("Check sum value is %lf \n",sum);
  	if(sum == 2851602432.000000){
  		printf("Check sum of image validated \n");
  	}
  	else{
  		printf("Check sum is wrong %lf \n",sum);
  		printf("Exiting program \n");
  		// exit(0);
  	}
}

void print_max_pool_plus_add_checksum(int batch, int channels, int height, int width, float *output_pointer){
	float sum = 0.0;
	for(int n= 0;n<batch;n++){
		for(int k = 0; k < channels; k++)
			for(int i = 0; i < height; i++){
	  			for(int j = 0; j < width; j++){
	  			{
	  				int index = n*channels*height*width + i*width + j + k*width*height;
	  				sum = sum + output_pointer[index];
	  			}
	  		}
	  	}
	}
  	printf("The checksum after the max_pool is %lf \n",sum);
}

float max_pool_plus_add_checksum(int batch, int channels, int height, int width, float *output_pointer){
	float sum = 0.0;
	for(int n= 0;n<batch;n++){
		for(int k = 0; k < channels; k++)
			for(int i = 0; i < height; i++){
	  			for(int j = 0; j < width; j++){
	  			{
	  				int index = n*channels*height*width + i*width + j + k*width*height;
	  				sum = sum + output_pointer[index];
	  			}
	  		}
	  	}
	}
	return sum;
}



void print_data(int batch, int channels,int height,int width, float *image_pointer){
	for(int b = 0;b<batch;b++){
		int b_offset  = b*channels*height*width;
		for(int c = 0; c < channels; c++){
	  		int c_offset = b_offset+ c*height*width;
	  		for(int i = 0; i< height; i++){
	  			for(int j = 0; j< width; j++){
	  				int index = c_offset + i*width + j;
	  				int cpu_value = image_pointer[index];
	  				printf(" %d ",cpu_value);
	  			}
	  			printf("\n");
	  		}
	  		printf("\n");
	  	}
	//printf("\n\n\n");
	}
}
