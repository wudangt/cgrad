double cpuSecond();
void initial_src1(int batch, int channels, int height, int width, double *image_pointer);
void initial_src2(int batch, int channels, int height, int width, double *image_pointer);
void validate_src_data(int batch, int channels, int height, int width, double *image_pointer);
void print_data(int batch, int channels,int height,int width, double *image_pointer);
double max_pool_plus_add_checksum(int batch, int channels, int height, int width, double *output_pointer);
void print_max_pool_plus_add_checksum(int batch, int channels, int height, int width, double *output_pointer);
