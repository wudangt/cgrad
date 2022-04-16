double cpuSecond();
void initial_src(int batch, int channels, int height, int width, float *image_pointer, int value);
void validate_src_data(int batch, int channels, int height, int width, float *image_pointer);
void print_data(int batch, int channels,int height,int width, float *image_pointer);
float max_pool_plus_add_checksum(int batch, int channels, int height, int width, float *output_pointer);
void print_max_pool_plus_add_checksum(int batch, int channels, int height, int width, float *output_pointer);
