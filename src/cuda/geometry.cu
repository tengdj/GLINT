#include <cuda.h>
#include "mygpu.h"
#include "cuda_util.h"
#include "../geometry/geometry.h"
#include "../util/query_context.h"
#include "../tracing/partitioner.h"

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
void reachability_cuda(const double *points, uint *zones, const uint *gridcheck, uint *ret, size_t num_checkes, size_t zone_size, double max_dist){

	// the objects in which grid need be processed
	int pairid = blockIdx.x*blockDim.x+threadIdx.x;
	if(pairid>=num_checkes){
		return;
	}

	uint pid = gridcheck[pairid*2];
	uint zid = gridcheck[pairid*2+1];

	uint size = *(zones+(zone_size+2)*zid+1);
	const uint *cur_pids = zones+(zone_size+2)*zid+2;
	double curx = *(points+pid*2);
	double cury = *(points+pid*2+1);
	//printf("%d %d %d\n",pid,gid,ret[pid]);
	for(uint i=0;i<size;i++){
		if(pid!=cur_pids[i]){
			double dist = distance(curx, cury,*(points+cur_pids[i]*2),*(points+cur_pids[i]*2+1));
			//printf("%d %d %d\n",pid,gid,ret[pid]);
			ret[pid] += dist<=max_dist;
		}
	}
	if(ret[pid]>1000){
		printf("%d\t%d\n",pid,ret[pid]);
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

double foo(Point *p1, Point *p2) {
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
	return dist;
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
	gpu->print();
	partition_info *pinfo = (partition_info *)ctx.target[0];
	uint *result = (uint *)ctx.target[1];
	Point *points = pinfo->points;

	size_t num_points = pinfo->num_objects;
	size_t num_zones = pinfo->cur_free_zone;
	size_t num_checkes = pinfo->num_grid_checkings;

	// space for the raw points data
	Point *d_points = (Point *)gpu->get_data(0, num_points*sizeof(Point));
	// space for the pids of all the grid zones
	uint *d_zones = (uint *)gpu->get_data(1, num_zones*(pinfo->zone_size+2)*sizeof(uint));
	// space for the pid-zid pairs
	uint *d_gridcheck = (uint *)gpu->get_data(2, 2*num_checkes*sizeof(uint));
	// space for the results in GPU
	uint *d_ret = (uint *)gpu->get_data(3, num_points*sizeof(uint));

	logt("allocating space", start);

	CUDA_SAFE_CALL(cudaMemcpy(d_points, points, num_points*sizeof(Point), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_zones, pinfo->buffer_zones, num_zones*(pinfo->zone_size+2)*sizeof(uint), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_gridcheck, pinfo->grid_checkings, 2*num_checkes*sizeof(uint), cudaMemcpyHostToDevice));

	cout<<(num_points*sizeof(Point)+num_zones*(pinfo->zone_size+2)*sizeof(uint)+2*num_checkes*sizeof(uint))/1024/1024<<endl;

	logt("copying data", start);
	// compute the reachability of objects in each partitions
	reachability_cuda<<<num_checkes/1024+1,1024>>>((double *)d_points, d_zones, d_gridcheck, d_ret, num_checkes, pinfo->zone_size, ctx.config.reach_distance);

	check_execution();
	cudaDeviceSynchronize();
	CUDA_SAFE_CALL(cudaMemcpy(result, d_ret, num_points*sizeof(uint), cudaMemcpyDeviceToHost));
	pthread_mutex_unlock(&gpu->lock);
	for(gpu_info *g:gpus){
		delete g;
	}
	ctx.found = 0;
	for(int i=0;i<num_points;i++){
		ctx.found += result[i];
	}
	logt("computing with GPU", start);
}

