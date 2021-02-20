#include <cuda.h>
#include "mygpu.h"
#include "cuda_util.h"
#include "../geometry/geometry.h"
#include "../util/query_context.h"
#include "../tracing/partitioner.h"
#include "../tracing/workbench.h"

//
__global__
void partition_cuda(workbench *bench){
	int pid = blockIdx.x*blockDim.x+threadIdx.x;
	if(pid>=bench->config->num_objects){
		return;
	}

	// search the tree to get in which grid
	uint curnode = 0;
	uint gid = 0;

	Point *p = bench->points+pid;
	while(true){
		int loc = (p->y>bench->schema[curnode].mid_y)*2
								+(p->x>bench->schema[curnode].mid_x);
		// is leaf
		if((bench->schema[curnode].children[loc]&1)){
			gid = bench->schema[curnode].children[loc]>>1;
			break;
		}else{
			curnode = bench->schema[curnode].children[loc]>>1;
		}
	}
	uint *cur_grid = bench->grids+(bench->config->grid_capacity+1)*gid;

	// insert current pid to proper memory space of the target gid
	uint cur_loc = atomicAdd(cur_grid,1);
	assert(cur_loc<bench->config->grid_capacity);
	*(cur_grid+1+cur_loc) = pid;
}

__device__
inline void lookup(workbench *bench, uint pid, uint curnode){

	Point *p = bench->points+pid;

	bool top = (p->y>bench->schema[curnode].mid_y-bench->config->y_buffer);
	bool bottom = (p->y<=bench->schema[curnode].mid_y+bench->config->y_buffer);
	bool left = (p->x<=bench->schema[curnode].mid_x+bench->config->x_buffer);
	bool right = (p->x>bench->schema[curnode].mid_x-bench->config->x_buffer);
	uint need_check = (bottom&&left)*1+(bottom&&right)*2+(top&&left)*4+(top&&right)*8;
	for(int i=0;i<4;i++){
		if((need_check>>i)&1){
			if((bench->schema[curnode].children[i]&1)){
				uint gid = bench->schema[curnode].children[i]>>1;
				assert(gid<bench->num_grids);
				uint offset = 0;
				while(offset<bench->grids[gid*(bench->config->grid_capacity+1)]){
					uint cu_index = atomicAdd(&bench->num_checking_units, 1);
					bench->checking_units[cu_index].pid = pid;
					bench->checking_units[cu_index].gid = gid;
					bench->checking_units[cu_index].offset = offset;
					//printf("%d\t%d\t%d\n",pid,gid,offset);
					offset += bench->config->zone_capacity;
				}
			}else{
				lookup(bench, pid, bench->schema[curnode].children[i]>>1);
			}
		}
	}
}

// with recursive call
__global__
void lookup_recursive_cuda(workbench *bench){
	int pid = blockIdx.x*blockDim.x+threadIdx.x;
	if(pid>=bench->config->num_objects){
		return;
	}
	lookup(bench,pid,0);
	return;
}

__global__
void initstack_cuda(workbench *bench){
	int pid = blockIdx.x*blockDim.x+threadIdx.x;
	if(pid>=bench->config->num_objects){
		return;
	}
	uint stack_index = atomicAdd(&bench->stack_index[0],1);
	assert(stack_index<bench->stack_capacity);
	bench->lookup_stack[0][stack_index*2] = pid;
	bench->lookup_stack[0][stack_index*2+1] = 0;
	printf("%d %d %d\n",stack_index,bench->lookup_stack[0][stack_index*2],bench->lookup_stack[0][stack_index*2+1]);
}

__global__
void lookup_cuda(workbench *bench, uint stack_id){

	int sid = blockIdx.x*blockDim.x+threadIdx.x;
	if(sid>=bench->stack_index[stack_id]){
		return;
	}

	uint pid = bench->lookup_stack[stack_id][sid*2];
	uint curnode = bench->lookup_stack[stack_id][sid*2+1];
	Point *p = bench->points+pid;

	// could be possibly in multiple children with buffers enabled
	bool top = (p->y>bench->schema[curnode].mid_y-bench->config->y_buffer);
	bool bottom = (p->y<=bench->schema[curnode].mid_y+bench->config->y_buffer);
	bool left = (p->x<=bench->schema[curnode].mid_x+bench->config->x_buffer);
	bool right = (p->x>bench->schema[curnode].mid_x-bench->config->x_buffer);
	uint need_check = (bottom&&left)*1+(bottom&&right)*2+(top&&left)*4+(top&&right)*8;
	for(int i=0;i<4;i++){
		if((need_check>>i)&1){
			if((bench->schema[curnode].children[i]&1)){
				uint gid = bench->schema[curnode].children[i]>>1;
				assert(gid<bench->num_grids);
				uint offset = 0;
				while(offset<bench->grids[gid*(bench->config->grid_capacity+1)]){
					uint cu_index = atomicAdd(&bench->num_checking_units, 1);
					assert(cu_index<bench->checking_units_capacity);
					bench->checking_units[cu_index].pid = pid;
					bench->checking_units[cu_index].gid = gid;
					bench->checking_units[cu_index].offset = offset;
					//printf("%d\t%d\t%d\n",pid,gid,offset);
					offset += bench->config->zone_capacity;
				}
			}else{
				uint stack_index = atomicAdd(&bench->stack_index[!stack_id],1);
				assert(stack_index<bench->stack_capacity);
				bench->lookup_stack[!stack_id][stack_index*2] = pid;
				bench->lookup_stack[!stack_id][stack_index*2+1] = bench->schema[curnode].children[i]>>1;
			}
		}
	}
	// reset the index to 0
	if(sid == 0){
		bench->stack_index[stack_id] = 0;
	}
}


__global__
void reachability_cuda(workbench *bench){

	// the objects in which grid need be processed
	int pairid = blockIdx.x*blockDim.x+threadIdx.x;
	if(pairid>=bench->num_checking_units){
		return;
	}

	double max_dist = bench->config->reach_distance;
	uint pid = bench->checking_units[pairid].pid;
	uint gid = bench->checking_units[pairid].gid;
	uint offset = bench->checking_units[pairid].offset;
	uint size = *(bench->grids+(bench->config->grid_capacity+1)*gid)-offset;

	if(size>bench->config->zone_capacity){
		size = bench->config->zone_capacity;
	}
	//printf("%d\t%d\t%d\t%d\n",pid,gid,offset,size);

	const uint *cur_pids = bench->grids+(bench->config->grid_capacity+1)*gid+1+offset;
	for(uint i=0;i<size;i++){
		if(pid!=cur_pids[i]){
			double dist = distance(bench->points[pid].x, bench->points[pid].y, bench->points[cur_pids[i]].x, bench->points[cur_pids[i]].y);
			if(dist<=max_dist){
				uint loc = atomicAdd(&bench->num_meeting, (uint)1);
				assert(loc<bench->meeting_capacity);
				bench->meetings[loc].pid1 = pid;
				bench->meetings[loc].pid2 = cur_pids[i];
			}
		}
	}
}


/*
 *
 * check the reachability of objects in a list of partitions
 * ctx.data contains the list of
 *
 * */
void process_with_gpu(workbench *bench){
	struct timeval start = get_cur_time();
	vector<gpu_info *> gpus = get_gpus();
	gpu_info *gpu = gpus[0];
	//gpu->print();
	assert(gpu);

	pthread_mutex_lock(&gpu->lock);
	cudaSetDevice(gpu->device_id);

	// use h_bench as a container to copy in and out GPU
	workbench *h_bench = new workbench(bench->config);
	h_bench->num_grids = bench->num_grids;

	// space for the raw points data
	h_bench->points = (Point *)gpu->get_data(0, bench->config->num_objects*sizeof(Point));
	// space for the pids of all the grids
	h_bench->grids = (uint *)gpu->get_data(1, bench->num_grids*(bench->config->grid_capacity+1)*sizeof(uint));
	// space for the pid-zid pairs
	h_bench->checking_units = (checking_unit *)gpu->get_data(2, bench->checking_units_capacity*sizeof(checking_unit));
	// space for the QTtree schema
	h_bench->schema = (QTSchema *)gpu->get_data(3, bench->num_nodes*sizeof(QTSchema));
	// space for processing stack
	h_bench->lookup_stack[0] = (uint *)gpu->get_data(4, bench->stack_capacity*2*sizeof(uint));
	h_bench->lookup_stack[1] = (uint *)gpu->get_data(5, bench->stack_capacity*2*sizeof(uint));
	h_bench->meetings = (meeting_unit *)gpu->get_data(6, bench->meeting_capacity*sizeof(meeting_unit));

	// space for the mapping of bench in GPU
	workbench *d_bench = (workbench *)gpu->get_data(7, sizeof(workbench));
	logt("allocating space %d MB", start,gpu->size_allocated()/1024/1024);

	struct timeval start_execute = get_cur_time();

	CUDA_SAFE_CALL(cudaMemcpy(h_bench->points, bench->points, bench->config->num_objects*sizeof(Point), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(h_bench->schema, bench->schema, bench->num_nodes*sizeof(QTSchema), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_bench, h_bench, sizeof(workbench), cudaMemcpyHostToDevice));
	logt("copying data", start);

	partition_cuda<<<bench->config->num_objects/1024+1,1024>>>(d_bench);
	check_execution();
	cudaDeviceSynchronize();
	logt("partition data", start);

	initstack_cuda<<<bench->config->num_objects/1024+1,1024>>>(d_bench);
	check_execution();
	cudaDeviceSynchronize();
	CUDA_SAFE_CALL(cudaMemcpy(h_bench, d_bench, sizeof(workbench), cudaMemcpyDeviceToHost));
	uint stack_id = 0;
	while(h_bench->stack_index[stack_id]>0){
		struct timeval tt = get_cur_time();
		lookup_cuda<<<h_bench->stack_index[stack_id]/1024+1,1024>>>(d_bench,stack_id);
		check_execution();
		cudaDeviceSynchronize();
		CUDA_SAFE_CALL(cudaMemcpy(h_bench, d_bench, sizeof(workbench), cudaMemcpyDeviceToHost));
		stack_id = !stack_id;
	}
	logt("lookup", start);

	// compute the reachability of objects in each partitions
	reachability_cuda<<<h_bench->num_checking_units/1024+1,1024>>>(d_bench);
	check_execution();
	cudaDeviceSynchronize();
	logt("computing reachability", start);
	logt("one round",start_execute);

	bench->num_checking_units = h_bench->num_checking_units;
	CUDA_SAFE_CALL(cudaMemcpy(bench->grids, h_bench->grids, bench->num_grids*(bench->config->grid_capacity+1)*sizeof(uint), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(bench->checking_units, h_bench->checking_units, h_bench->num_checking_units*sizeof(checking_unit), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(bench->meetings, h_bench->meetings, h_bench->num_meeting*sizeof(meeting_unit), cudaMemcpyDeviceToHost));


	h_bench->grids = NULL;
	h_bench->checking_units = NULL;
	h_bench->schema = NULL;
	h_bench->lookup_stack[0] = NULL;
	h_bench->lookup_stack[1] = NULL;
	h_bench->meetings = NULL;
	delete h_bench;
	pthread_mutex_unlock(&gpu->lock);
	for(gpu_info *g:gpus){
		delete g;
	}
}
