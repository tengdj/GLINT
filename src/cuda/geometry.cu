#include <cuda.h>
#include "mygpu.h"
#include "cuda_util.h"
#include "../geometry/geometry.h"
#include "../util/query_context.h"
#include "../tracing/partitioner.h"



//
__global__
void partition_cuda(partition_info *pinfo){
	int pid = blockIdx.x*blockDim.x+threadIdx.x;
	if(pid>=pinfo->num_objects){
		return;
	}

	// search the tree to get in which grid
	uint curoff = 0;
	uint gid = 0;

	Point *p = pinfo->points+pid;
	while(true){
		int loc = (p->y>pinfo->schema[curoff].mid_y)*2+(p->x>pinfo->schema[curoff].mid_x);
		// is leaf
		if((pinfo->schema[curoff].children[loc]&1)){
			gid = pinfo->schema[curoff].children[loc]>>1;
			break;
		}else{
			curoff = pinfo->schema[curoff].children[loc]>>1;
		}
	}
	uint *cur_grid = pinfo->grids+(pinfo->grid_capacity+1)*gid;

	// insert current pid to proper memory space of the target gid
	uint cur_loc = atomicAdd(cur_grid,1);
	*(cur_grid+1+cur_loc) = pid;
}


__global__
void reachability_cuda(const partition_info *pinfo, uint *ret, double max_dist){

	// the objects in which grid need be processed
	int pairid = blockIdx.x*blockDim.x+threadIdx.x;
	if(pairid>=pinfo->num_checking_units){
		return;
	}

	uint pid = pinfo->checking_units[pairid].pid;
	uint gid = pinfo->checking_units[pairid].gid;
	uint offset = pinfo->checking_units[pairid].offset;
	uint size = *(pinfo->grids+(pinfo->grid_capacity+1)*gid)-offset;

	if(size>pinfo->unit_size){
		size = pinfo->unit_size;
	}
	//printf("%d %d %d %d\n",pid,gid,gridcheck[pairid].offset,size);

	const uint *cur_pids = pinfo->grids+(pinfo->grid_capacity+1)*gid+1+offset;
	for(uint i=0;i<size;i++){
		if(pid!=cur_pids[i]){
			double dist = distance(pinfo->points[pid].x, pinfo->points[pid].y, pinfo->points[cur_pids[i]].x, pinfo->points[cur_pids[i]].y);
			atomicAdd(ret+pid, dist<=max_dist);
		}
	}
//	if(ret[pid]>1000){
//		printf("%d\t%d\n",pid,ret[pid]);
//	}
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
	//gpu->print();
	assert(gpu);

	pthread_mutex_lock(&gpu->lock);
	cudaSetDevice(gpu->device_id);

	partition_info *pinfo = (partition_info *)ctx.target[0];

	// space for the raw points data
	Point *d_points = (Point *)gpu->get_data(0, pinfo->num_objects*sizeof(Point));
	// space for the pids of all the grids
	uint *d_grids = (uint *)gpu->get_data(1, pinfo->num_grids*(pinfo->grid_capacity+1)*sizeof(uint));
	// space for the pid-zid pairs
	checking_unit *d_gridcheck = (checking_unit *)gpu->get_data(2, pinfo->num_checking_units*sizeof(checking_unit));
	// space for the QTtree schema
	QTSchema *d_schema = (QTSchema *)gpu->get_data(3, pinfo->num_nodes*sizeof(QTSchema));
	// space for the mapping of pinfo in GPU
	partition_info *d_pinfo = (partition_info *)gpu->get_data(4, sizeof(partition_info));
	// space for the results in GPU
	uint *d_ret = (uint *)gpu->get_data(5, pinfo->num_objects*sizeof(uint));
	logt("allocating space", start);

	CUDA_SAFE_CALL(cudaMemcpy(d_points, pinfo->points, pinfo->num_objects*sizeof(Point), cudaMemcpyHostToDevice));
	//CUDA_SAFE_CALL(cudaMemcpy(d_grids, pinfo->grids, pinfo->num_grids*(pinfo->grid_capacity+1)*sizeof(uint), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_gridcheck, pinfo->checking_units, pinfo->num_checking_units*sizeof(checking_unit), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_schema, pinfo->schema, pinfo->num_nodes*sizeof(QTSchema), cudaMemcpyHostToDevice));

	// use pinfo as a container to copy into GPU
	do{
		Point *h_points = pinfo->points;
		uint *h_grids = pinfo->grids;
		checking_unit *h_gridcheck = pinfo->checking_units;
		QTSchema *h_schema = pinfo->schema;

		pinfo->points = d_points;
		pinfo->grids = d_grids;
		pinfo->checking_units = d_gridcheck;
		pinfo->schema = d_schema;
		CUDA_SAFE_CALL(cudaMemcpy(d_pinfo, pinfo, sizeof(partition_info), cudaMemcpyHostToDevice));
		pinfo->points = h_points;
		pinfo->grids = h_grids;
		pinfo->checking_units = h_gridcheck;
		pinfo->schema = h_schema;
	}while(false);
	logt("copying data", start);

	//cout<<(num_points*sizeof(Point)+num_zones*(pinfo->zone_size+2)*sizeof(uint)+2*num_checkes*sizeof(uint))/1024/1024<<endl;

	partition_cuda<<<pinfo->num_objects/1024+1,1024>>>(d_pinfo);
	check_execution();
	cudaDeviceSynchronize();
	logt("partition data", start);

	// compute the reachability of objects in each partitions
	reachability_cuda<<<pinfo->num_checking_units/1024+1,1024>>>(d_pinfo, d_ret, ctx.config.reach_distance);

	check_execution();
	cudaDeviceSynchronize();
	CUDA_SAFE_CALL(cudaMemcpy(ctx.target[1], d_ret, pinfo->num_objects*sizeof(uint), cudaMemcpyDeviceToHost));
	pthread_mutex_unlock(&gpu->lock);
	for(gpu_info *g:gpus){
		delete g;
	}
	logt("computing with GPU", start);
}
