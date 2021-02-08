/*
 *
 * with some common gpu related operations
 * */

#include <cuda.h>
#include "mygpu.h"
#include "cuda_util.h"
#include "../util/util.h"

using namespace std;



vector<gpu_info *> get_gpus(){
	vector<gpu_info *> gpus;
	int num_gpus = 0;
	cudaGetDeviceCount(&num_gpus);
	for (int i = 0; i < num_gpus; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		gpu_info *info = new gpu_info();
		info->busy = false;
		info->mem_size = prop.totalGlobalMem/1024/1024*4/5;
		info->device_id = i;
		// we allocate 2G mem for each gpu
//		if(info->mem_size>2048){
//			info->mem_size = 2048;
//		}
		gpus.push_back(info);
	}
	return gpus;
}

void print_gpus(){
	int num_gpus = 0;
	cudaGetDeviceCount(&num_gpus);
	for (int i = 0; i < num_gpus; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		log("Device ID: %d", i);
		log("  Device name: %s", prop.name);
		log("  Memory Clock Rate (KHz): %d", prop.memoryClockRate);
		log("  Memory Bus Width (bits): %d", prop.memoryBusWidth);
		log("  Peak Memory Bandwidth (GB/s): %f",
				2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
		log("  Memory size (MB): %ld", prop.totalGlobalMem/1024/1024);
	}
}


void gpu_info::init(){
	for(int i=0;i<MAX_DATA_SPACE;i++){
		data_size[i] = 0;
		d_data[i] = NULL;
	}
}

void *gpu_info::get_data(int did, size_t ss){
	assert(did<MAX_DATA_SPACE);
	cudaSetDevice(this->device_id);
	if(this->d_data[did]&&this->data_size[did]<ss){
		CUDA_SAFE_CALL(cudaFree(this->d_data[did]));
		this->data_size[did] = 0;
		this->d_data[did] = NULL;
	}
	if(!this->d_data[did]){
		CUDA_SAFE_CALL(cudaMalloc((void **)&d_data[did], ss));
		assert(this->d_data[did]);
		this->data_size[did] = ss;
	}
	return this->d_data[did];
}

gpu_info::~gpu_info(){
	cudaSetDevice(this->device_id);
	for(int i=0;i<MAX_DATA_SPACE;i++){
		if(d_data[i]){
			CUDA_SAFE_CALL(cudaFree(this->d_data[i]));
			d_data[i] = NULL;
			data_size[i] = 0;
		}
	}
}

