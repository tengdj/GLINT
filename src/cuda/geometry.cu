#include <cuda.h>
#include "mygpu.h"
#include "cuda_util.h"
#include "../geometry/geometry.h"
#include "../util/query_context.h"

// return the distance of two segments

const static double degree_per_meter_latitude_cuda = 360.0/(40076.0*1000);

__device__
inline double degree_per_meter_longitude_cuda(double latitude){
	return 360.0/(sin((90-abs(latitude))*PI/180)*40076.0*1000.0);
}

__device__
inline double distance(const double x1, const double y1, const double x2, const double y2){
	double dx = x1-x2;
	double dy = y1-y2;
	dx = dx/degree_per_meter_longitude_cuda(y1);
	dy = dy/degree_per_meter_latitude_cuda;
	return sqrt(dx*dx+dy*dy);
}

__global__
void reachability_cuda(const double *points, const uint *gridassign, uint *pids, const offset_size *os, uint *ret, size_t num_points, double max_dist){

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
	for(uint i=0;i<os[gid].size;i++){
		if(pid!=cur_pids[i]){
			double dist = distance(*(points+pid*2),*(points+pid*2+1),*(points+cur_pids[i]*2),*(points+cur_pids[i]*2+1));
			ret[pid] += dist<=max_dist;
		}
	}
}


query_context partition_with_gpu(Point *points, size_t num_objects, offset_size *os){

	query_context ctx;

	return ctx;
}

__global__ void mykernel(Point *p1, Point *p2, double *dist) {
	*dist = distance(p1->x,p1->y,p2->x,p2->y);
	printf("gpu %f\n",*dist);
  //*addr += 10;
  //atomicAdd(addr, 10);       // only available on devices with compute capability 6.x
}

int foo(Point *p1, Point *p2) {
	Point *d_p1,*d_p2;
	double *d_dist;
	cudaMallocManaged(&d_p1, sizeof(Point));
	cudaMallocManaged(&d_p2, sizeof(Point));
	cudaMallocManaged(&d_dist, sizeof(double));

	CUDA_SAFE_CALL(cudaMemcpy(d_p1, p1, sizeof(Point), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_p2, p2, sizeof(Point), cudaMemcpyHostToDevice));

	mykernel<<<1,1>>>(d_p1, d_p2, d_dist);
	double dist = 0;
	CUDA_SAFE_CALL(cudaMemcpy(&dist, d_dist, sizeof(double), cudaMemcpyDeviceToHost));
	int ret = 0;
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
	uint *d_ret = (uint *)gpu->get_data(3, sizeof(uint)*num_points);

	uint *d_gridassign = (uint *)gpu->get_data(4, sizeof(uint)*num_points);
	logt("allocating space", start);

	CUDA_SAFE_CALL(cudaMemcpy(d_points, points, num_points*sizeof(Point), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_gridassign, grid_resignment, num_points*sizeof(uint), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_pids, pids, num_objects*sizeof(uint), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_os, os, num_grids*sizeof(offset_size), cudaMemcpyHostToDevice));
	logt("copying data", start);
	// compute the reachability of objects in each partitions
	reachability_cuda<<<num_points/1024+1,1024>>>((double *)d_points, d_gridassign, d_pids, d_os, d_ret, num_points, ctx.config.reach_distance);

	check_execution();
	cudaDeviceSynchronize();
	CUDA_SAFE_CALL(cudaMemcpy(result, d_ret, num_points*sizeof(uint), cudaMemcpyDeviceToHost));
	pthread_mutex_unlock(&gpu->lock);
	for(gpu_info *g:gpus){
		delete g;
	}
	for(int i=0;i<num_grids;i++){
		ctx.found += result[i];
	}
	logt("computing with GPU", start);
}

