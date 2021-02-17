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
	if(pid>=pinfo->config.num_objects){
		return;
	}

	// search the tree to get in which grid
	uint curnode = 0;
	uint gid = 0;

	Point *p = pinfo->points+pid;
	while(true){
		int loc = (p->y>pinfo->schema[curnode].mid_y)*2+(p->x>pinfo->schema[curnode].mid_x);
		// is leaf
		if((pinfo->schema[curnode].children[loc]&1)){
			gid = pinfo->schema[curnode].children[loc]>>1;
			break;
		}else{
			curnode = pinfo->schema[curnode].children[loc]>>1;
		}
	}
	uint *cur_grid = pinfo->grids+(pinfo->config.grid_capacity+1)*gid;

	// insert current pid to proper memory space of the target gid
	uint cur_loc = atomicAdd(cur_grid,1);
	assert(cur_loc<pinfo->config.grid_capacity);
	*(cur_grid+1+cur_loc) = pid;
}

//__device__
//inline void lookup(partition_info *pinfo, uint pid, uint curnode){
//
//	Point *p = pinfo->points+pid;
//
//	bool top = (p->y>pinfo->schema[curnode].mid_y-pinfo->config.y_buffer);
//	bool bottom = (p->y<=pinfo->schema[curnode].mid_y+pinfo->config.y_buffer);
//	bool left = (p->x<=pinfo->schema[curnode].mid_x+pinfo->config.x_buffer);
//	bool right = (p->x>pinfo->schema[curnode].mid_x-pinfo->config.x_buffer);
//	uint need_check = (bottom&&left)*1+(bottom&&right)*2+(top&&left)*4+(top&&right)*8;
//	for(int i=0;i<4;i++){
//		if((need_check>>i)&1){
//			if((pinfo->schema[curnode].children[i]&1)){
//				uint gid = pinfo->schema[curnode].children[i]>>1;
//				assert(gid<pinfo->num_grids);
//				uint offset = 0;
//				while(offset<pinfo->grids[gid*(pinfo->config.grid_capacity+1)]){
//					uint cu_index = atomicAdd(&pinfo->num_checking_units, 1);
//					pinfo->checking_units[cu_index].pid = pid;
//					pinfo->checking_units[cu_index].gid = gid;
//					pinfo->checking_units[cu_index].offset = offset;
//					//printf("%d\t%d\t%d\n",pid,gid,offset);
//					offset += pinfo->config.zone_capacity;
//				}
//			}else{
//				lookup(pinfo, pid, pinfo->schema[curnode].children[i]>>1);
//			}
//		}
//	}
//}
//
//// with recursive call
//__global__
//void lookup_recursive_cuda(partition_info *pinfo){
//	int pid = blockIdx.x*blockDim.x+threadIdx.x;
//	if(pid>=pinfo->config.num_objects){
//		return;
//	}
//	lookup(pinfo,pid,0);
//	return;
//}

__global__
void initstack_cuda(partition_info *pinfo){
	int pid = blockIdx.x*blockDim.x+threadIdx.x;
	if(pid>=pinfo->config.num_objects){
		return;
	}
	uint stack_index = atomicAdd(&pinfo->stack_index[0],1);
	pinfo->lookup_stack[0][stack_index*2] = pid;
	pinfo->lookup_stack[0][stack_index*2+1] = 0;
}

__global__
void lookup_cuda(partition_info *pinfo, uint stack_id, uint stack_size){

	int sid = blockIdx.x*blockDim.x+threadIdx.x;
	if(sid>=stack_size){
		return;
	}

	uint pid = pinfo->lookup_stack[stack_id][sid*2];
	uint curnode = pinfo->lookup_stack[stack_id][sid*2+1];
	Point *p = pinfo->points+pid;

	// could be possibly in multiple children with buffers enabled
	bool top = (p->y>pinfo->schema[curnode].mid_y-pinfo->config.y_buffer);
	bool bottom = (p->y<=pinfo->schema[curnode].mid_y+pinfo->config.y_buffer);
	bool left = (p->x<=pinfo->schema[curnode].mid_x+pinfo->config.x_buffer);
	bool right = (p->x>pinfo->schema[curnode].mid_x-pinfo->config.x_buffer);
	uint need_check = (bottom&&left)*1+(bottom&&right)*2+(top&&left)*4+(top&&right)*8;
	for(int i=0;i<4;i++){
		if((need_check>>i)&1){
			if((pinfo->schema[curnode].children[i]&1)){
				uint gid = pinfo->schema[curnode].children[i]>>1;
				assert(gid<pinfo->num_grids);
				uint offset = 0;
				while(offset<pinfo->grids[gid*(pinfo->config.grid_capacity+1)]){
					uint cu_index = atomicAdd(&pinfo->num_checking_units, 1);
					assert(cu_index<pinfo->checking_units_capacity);
					pinfo->checking_units[cu_index].pid = pid;
					pinfo->checking_units[cu_index].gid = gid;
					pinfo->checking_units[cu_index].offset = offset;
					//printf("%d\t%d\t%d\n",pid,gid,offset);
					offset += pinfo->config.zone_capacity;
				}
			}else{
				uint stack_index = atomicAdd(&pinfo->stack_index[!stack_id],1);
				pinfo->lookup_stack[!stack_id][stack_index*2] = pid;
				pinfo->lookup_stack[!stack_id][stack_index*2+1] = pinfo->schema[curnode].children[i]>>1;
			}
		}
	}
	if(sid == 0){
		pinfo->stack_index[stack_id] = 0;
	}
}


__global__
void reachability_cuda(const partition_info *pinfo, uint *ret){

	// the objects in which grid need be processed
	int pairid = blockIdx.x*blockDim.x+threadIdx.x;
	if(pairid>=pinfo->num_checking_units){
		return;
	}

	double max_dist = pinfo->config.reach_distance;
	uint pid = pinfo->checking_units[pairid].pid;
	uint gid = pinfo->checking_units[pairid].gid;
	uint offset = pinfo->checking_units[pairid].offset;
	uint size = *(pinfo->grids+(pinfo->config.grid_capacity+1)*gid)-offset;

	if(size>pinfo->config.zone_capacity){
		size = pinfo->config.zone_capacity;
	}
	//printf("%d\t%d\t%d\t%d\n",pid,gid,offset,size);

	const uint *cur_pids = pinfo->grids+(pinfo->config.grid_capacity+1)*gid+1+offset;
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
	// use h_pinfo as a container to copy in and out GPU
	partition_info *h_pinfo = new partition_info(pinfo->config);
	h_pinfo->num_grids = pinfo->num_grids;

	// space for the raw points data
	h_pinfo->points = (Point *)gpu->get_data(0, pinfo->config.num_objects*sizeof(Point));
	// space for the pids of all the grids
	h_pinfo->grids = (uint *)gpu->get_data(1, pinfo->num_grids*(pinfo->config.grid_capacity+1)*sizeof(uint));
	// space for the pid-zid pairs
	h_pinfo->checking_units = (checking_unit *)gpu->get_data(2, pinfo->checking_units_capacity*sizeof(checking_unit));
	// space for the QTtree schema
	h_pinfo->schema = (QTSchema *)gpu->get_data(3, pinfo->num_nodes*sizeof(QTSchema));
	// space for processing stack
	h_pinfo->lookup_stack[0] = (uint *)gpu->get_data(4, 2*2*pinfo->config.num_objects*sizeof(uint));
	h_pinfo->lookup_stack[1] = (uint *)gpu->get_data(5, 2*2*pinfo->config.num_objects*sizeof(uint));
	// space for the mapping of pinfo in GPU
	partition_info *d_pinfo = (partition_info *)gpu->get_data(6, sizeof(partition_info));
	// space for the results in GPU
	uint *d_ret = (uint *)gpu->get_data(7, pinfo->config.num_objects*sizeof(uint));
	logt("allocating space %d MB", start,gpu->size_allocated()/1024/1024);

	struct timeval start_execute = get_cur_time();

	CUDA_SAFE_CALL(cudaMemcpy(h_pinfo->points, pinfo->points, pinfo->config.num_objects*sizeof(Point), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(h_pinfo->schema, pinfo->schema, pinfo->num_nodes*sizeof(QTSchema), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_pinfo, h_pinfo, sizeof(partition_info), cudaMemcpyHostToDevice));
	logt("copying data", start);

	partition_cuda<<<pinfo->config.num_objects/1024+1,1024>>>(d_pinfo);
	check_execution();
	cudaDeviceSynchronize();
	logt("partition data", start);

	initstack_cuda<<<pinfo->config.num_objects/1024+1,1024>>>(d_pinfo);
	CUDA_SAFE_CALL(cudaMemcpy(h_pinfo, d_pinfo, sizeof(partition_info), cudaMemcpyDeviceToHost));
	uint stack_id = 0;
	while(h_pinfo->stack_index[stack_id]>0){
		struct timeval tt = get_cur_time();
		lookup_cuda<<<h_pinfo->stack_index[stack_id]/1024+1,1024>>>(d_pinfo,stack_id,h_pinfo->stack_index[stack_id]);
		check_execution();
		cudaDeviceSynchronize();
		CUDA_SAFE_CALL(cudaMemcpy(h_pinfo, d_pinfo, sizeof(partition_info), cudaMemcpyDeviceToHost));
		stack_id = !stack_id;
	}
	logt("lookup", start);

	// compute the reachability of objects in each partitions
	reachability_cuda<<<h_pinfo->num_checking_units/1024+1,1024>>>(d_pinfo, d_ret);
	check_execution();
	cudaDeviceSynchronize();
	logt("computing reachability", start);
	logt("one round",start_execute);

	pinfo->num_checking_units = h_pinfo->num_checking_units;
	CUDA_SAFE_CALL(cudaMemcpy(ctx.target[1], d_ret, pinfo->config.num_objects*sizeof(uint), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(pinfo->grids, h_pinfo->grids, pinfo->num_grids*(pinfo->config.grid_capacity+1)*sizeof(uint), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(pinfo->checking_units, h_pinfo->checking_units, h_pinfo->num_checking_units*sizeof(checking_unit), cudaMemcpyDeviceToHost));

	h_pinfo->grids = NULL;
	h_pinfo->checking_units = NULL;
	h_pinfo->schema = NULL;
	delete h_pinfo;
	pthread_mutex_unlock(&gpu->lock);
	for(gpu_info *g:gpus){
		delete g;
	}
}
