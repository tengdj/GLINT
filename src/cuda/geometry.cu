#include <cuda.h>
#include "mygpu.h"
#include "cuda_util.h"
#include "../geometry/geometry.h"
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
void reachability_cuda(const double *points, uint *pids, const offset_size *os, uint *ret, size_t num_grids, double max_dist){

	// the objects in which grid need be processed
	int gid = blockIdx.x*blockDim.x+threadIdx.x;
	if(gid>=num_grids){
		return;
	}
	ret[gid] = 0;

	if(os[gid].size<=1){
		return;
	}
	const uint *cur_pids = pids+os[gid].offset;
	double sqrt_max_dist = max_dist*max_dist;
	for(uint i=0;i<os[gid].size-1;i++){
		for(uint j=i+1;j<os[gid].size;j++){
			ret[gid] += distance(points+cur_pids[i]*2, points+cur_pids[j]*2)<=sqrt_max_dist;
		}
	}
}

__global__
void reachability_cuda2(const double *points, const uint *gridassign, uint *pids, const offset_size *os, uint *ret, size_t num_points, double max_dist){

	// the objects in which grid need be processed
	int pid = blockIdx.x*blockDim.x+threadIdx.x;
	if(pid>=num_points){
		return;
	}

	uint gid = gridassign[pid];
	if(os[gid].size<=1){
		return;
	}
	const uint *cur_pids = pids+os[gid].offset;
	double sqrt_max_dist = max_dist*max_dist;

	for(uint i=0;i<os[gid].size;i++){
		if(pid!=cur_pids[i]){
			ret[gid] += distance(points+cur_pids[i]*2, points+pid*2)<=sqrt_max_dist;
		}
	}
}


query_context partition_with_gpu(Point *points, size_t num_objects, offset_size *os){

	query_context ctx;

	return ctx;
}

__global__ void mykernel(int *addr) {
	*addr += 10;
  //atomicAdd(addr, 10);       // only available on devices with compute capability 6.x
}

int foo() {
	int *addr;
	cudaMallocManaged(&addr, 4);
	mykernel<<<10,1024>>>(addr);
	__sync_fetch_and_add(addr, 10);  // CPU atomic operation
	int ret = 0;
	CUDA_SAFE_CALL(cudaMemcpy(&ret, addr, 4, cudaMemcpyDeviceToHost));
	return ret;
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

	Point *points = (Point *)ctx.target[0];
	uint *pids = (uint *)ctx.target[1];
	offset_size *os = (offset_size *)ctx.target[2];
	uint *result = (uint *)ctx.target[3];
	uint *grid_resignment = (uint *)ctx.target[4];

	size_t num_points = ctx.target_length[0];
	size_t num_objects = ctx.target_length[1];
	size_t num_grids = ctx.target_length[2];

	// space for the raw points data
	Point *d_points = (Point *)gpu->get_data(0, sizeof(Point)*num_points);
	// space for the pids of all the grids
	uint *d_pids = (uint *)gpu->get_data(1, num_objects*sizeof(uint));
	// space for the offset and size information in GPU
	offset_size *d_os = (offset_size *)gpu->get_data(2, sizeof(offset_size)*num_grids);
	// space for the results in GPU
	uint *d_ret = (uint *)gpu->get_data(3, sizeof(uint)*num_grids);

	uint *d_gridassign = (uint *)gpu->get_data(4, sizeof(uint)*num_points);
	logt("allocating space", start);

	CUDA_SAFE_CALL(cudaMemcpy(d_points, points, num_points*sizeof(Point), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_gridassign, grid_resignment, num_points*sizeof(uint), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_pids, pids, num_objects*sizeof(uint), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_os, os, num_grids*sizeof(offset_size), cudaMemcpyHostToDevice));
	logt("copying data", start);
	// compute the reachability of objects in each partitions
	reachability_cuda2<<<num_points/1024+1,1024>>>((double *)d_points, d_gridassign, d_pids, d_os, d_ret, num_points, ctx.config.reach_distance);

	check_execution();
	cudaDeviceSynchronize();
	CUDA_SAFE_CALL(cudaMemcpy(result, d_ret, num_grids*sizeof(uint), cudaMemcpyDeviceToHost));
	pthread_mutex_unlock(&gpu->lock);
	for(gpu_info *g:gpus){
		delete g;
	}
	for(int i=0;i<num_grids;i++){
		ctx.found += result[i];
	}
	logt("computing with GPU", start);
}

