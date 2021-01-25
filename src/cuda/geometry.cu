#include <cuda.h>
#include "mygpu.h"
#include "cuda_util.h"
#include "../util/query_context.h"

// return the distance of two segments

#define degree_per_kilometer_latitude_cuda 360.0/40076.0;

__device__
inline double degree_per_kilometer_longitude_cuda(double latitude){
	return 360.0/(sin((90-abs(latitude))*PI/180)*40076);
}

__device__
inline double distance(const double *point1, const double *point2){
	double dx = point1[0]-point2[0];
	double dy = point1[1]-point2[1];
	dx = dx/degree_per_kilometer_longitude_cuda(point1[1]);
	dy = dy/degree_per_kilometer_latitude_cuda;
	return dx*dx+dy*dy;
}

__global__
void reachability_cuda(const double *points, const uint *offset_size, int *ret, size_t num_grids, double max_dist){

	// the objects in which grid need be processed
	int grid_id = blockIdx.x*blockDim.x+threadIdx.x;
	if(grid_id>=num_grids){
		return;
	}
	ret[grid_id] = 0;
	uint offset = offset_size[grid_id*2];
	uint size = offset_size[grid_id*2+1];

	if(size==1){
		return;
	}
	const double *cur_points = points+offset*2;
	for(uint i=0;i<size-1;i++){
		for(uint j=i+1;j<size;j++){
			ret[grid_id] += distance(cur_points+i*2, cur_points+j*2)<=max_dist;
		}
	}
}


/*
 *
 * check the reachability of objects in a list of partitions
 * ctx->data contains the list of
 *
 * */
void process_with_gpu(gpu_info *gpu, query_context *ctx){
	pthread_mutex_lock(&gpu->lock);
	assert(gpu);
	cudaSetDevice(gpu->device_id);
	struct timeval start = get_cur_time();

	// space for the results in GPU
	int *d_ret = gpu->get_result(sizeof(int)*ctx->config.num_grids);
	// space for the offset and size information in GPU
	uint *d_os = gpu->get_os(sizeof(uint)*ctx->config.num_grids*2);
	double *d_partition = gpu->get_data(ctx->config.num_objects*2*sizeof(double));
	CUDA_SAFE_CALL(cudaMemcpy(d_partition, ctx->data, ctx->config.num_objects*2*sizeof(double), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_os, ctx->offset_size, ctx->config.num_grids*2*sizeof(uint), cudaMemcpyHostToDevice));
	logt("allocating data", start);
	// compute the reachability of objects in each partitions
	reachability_cuda<<<ctx->config.num_grids/1024+1,1024>>>(d_partition, d_os, d_ret, ctx->config.num_grids, ctx->config.reach_threshold);

	check_execution();
	cudaDeviceSynchronize();
	CUDA_SAFE_CALL(cudaMemcpy(ctx->result, d_ret, ctx->config.num_grids*sizeof(int), cudaMemcpyDeviceToHost));
	pthread_mutex_unlock(&gpu->lock);
	logt("computations", start);
}

