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
	max_dist *= max_dist;
	for(uint i=0;i<size-1;i++){
		for(uint j=i+1;j<size;j++){
			ret[grid_id] += distance(cur_points+i*2, cur_points+j*2)<=max_dist;
		}
	}
}


/*
 *
 * check the reachability of objects in a list of partitions
 * ctx.data contains the list of
 *
 * */
void process_with_gpu(query_context &ctx){
	struct timeval start = get_cur_time();

	vector<gpu_info *> gpus = get_gpus();
	gpu_info *gpu = gpus[0];

	pthread_mutex_lock(&gpu->lock);
	assert(gpu);
	cudaSetDevice(gpu->device_id);

	uint *offset_size = (uint *)ctx.target[1];
	double *points = (double *)ctx.target[0];
	int *result = (int *)ctx.target[2];
	size_t num_grids = ctx.counter;
	size_t num_objects = offset_size[2*num_grids-2]+offset_size[2*num_grids-1];
	cout<<num_grids<<" "<<num_objects<<endl;

	// space for the results in GPU
	int *d_ret = gpu->get_result(sizeof(int)*num_grids);
	// space for the offset and size information in GPU
	uint *d_os = gpu->get_os(sizeof(uint)*num_grids*2);
	double *d_partition = gpu->get_data(num_objects*2*sizeof(double));
	CUDA_SAFE_CALL(cudaMemcpy(d_partition, points, num_objects*2*sizeof(double), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_os, offset_size, num_grids*2*sizeof(uint), cudaMemcpyHostToDevice));
	logt("allocating data", start);
	// compute the reachability of objects in each partitions
	reachability_cuda<<<num_grids/1024+1,1024>>>(d_partition, d_os, d_ret, num_grids, ctx.config.reach_distance);

	check_execution();
	cudaDeviceSynchronize();
	CUDA_SAFE_CALL(cudaMemcpy(result, d_ret, num_grids*sizeof(int), cudaMemcpyDeviceToHost));
	pthread_mutex_unlock(&gpu->lock);
	for(gpu_info *g:gpus){
		delete g;
	}
	for(int i=0;i<num_grids;i++){
		ctx.found += result[i];
	}
	logt("computing with GPU", start);
}

