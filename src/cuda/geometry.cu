#include <cuda.h>
#include "mygpu.h"
#include "cuda_util.h"
#include "../geometry/geometry.h"
#include "../util/query_context.h"
#include "../tracing/partitioner.h"
#include "../tracing/workbench.h"



//__device__
//inline void lookup(workbench *bench, uint pid, uint curnode){
//
//	Point *p = bench->points+pid;
//
//	bool top = (p->y>bench->schema[curnode].mid_y-bench->config->y_buffer);
//	bool bottom = (p->y<=bench->schema[curnode].mid_y+bench->config->y_buffer);
//	bool left = (p->x<=bench->schema[curnode].mid_x+bench->config->x_buffer);
//	bool right = (p->x>bench->schema[curnode].mid_x-bench->config->x_buffer);
//	uint need_check = (bottom&&left)*1+(bottom&&right)*2+(top&&left)*4+(top&&right)*8;
//	for(int i=0;i<4;i++){
//		if((need_check>>i)&1){
//			if((bench->schema[curnode].children[i]&1)){
//				uint gid = bench->schema[curnode].children[i]>>1;
//				assert(gid<bench->num_grids);
//				uint offset = 0;
//				while(offset<bench->grids[gid*(bench->config->grid_capacity+1)]){
//					uint cu_index = atomicAdd(&bench->unit_lookup_counter, 1);
//					bench->unit_lookup[cu_index].pid = pid;
//					bench->unit_lookup[cu_index].gid = gid;
//					bench->unit_lookup[cu_index].offset = offset;
//					//printf("%d\t%d\t%d\n",pid,gid,offset);
//					offset += bench->config->zone_capacity;
//				}
//			}else{
//				lookup(bench, pid, bench->schema[curnode].children[i]>>1);
//			}
//		}
//	}
//}
//
//// with recursive call
//__global__
//void lookup_recursive_cuda(workbench *bench){
//	int pid = blockIdx.x*blockDim.x+threadIdx.x;
//	if(pid>=bench->config->num_objects){
//		return;
//	}
//	lookup(bench,pid,0);
//	return;
//}


////  partition with cuda
//__global__
//void partition_cuda(workbench *bench){
//	int pid = blockIdx.x*blockDim.x+threadIdx.x;
//	if(pid>=bench->config->num_objects){
//		return;
//	}
//
//	// search the tree to get in which grid
//	uint curnode = 0;
//	uint gid = 0;
//
//	Point *p = bench->points+pid;
//	while(true){
//		int loc = (p->y>bench->schema[curnode].mid_y)*2
//								+(p->x>bench->schema[curnode].mid_x);
//		uint child_offset = bench->schema[curnode].children[loc];
//		// is leaf
//		if(bench->schema[child_offset].isleaf){
//			gid = bench->schema[child_offset].node_id;
//			break;
//		}
//		curnode = child_offset;
//	}
//	uint *cur_grid = bench->grids+(bench->config->grid_capacity+1)*gid;
//	uint glid = atomicAdd(&bench->unit_lookup_counter,1);
//	bench->unit_lookup[glid].pid = pid;
//	bench->unit_lookup[glid].gid = gid;
//	bench->unit_lookup[glid].offset = 0;
//	// need also check the upper and right nodes
//	if(p->x+bench->config->x_buffer>bench->schema[curnode].mbr.high[0]||
//	   p->y+bench->config->y_buffer>bench->schema[curnode].mbr.high[1]){
//		uint stack_index = atomicAdd(&bench->stack_index[0],1);
//		assert(stack_index<bench->stack_capacity);
//		bench->lookup_stack[0][stack_index*2] = pid;
//		bench->lookup_stack[0][stack_index*2+1] = 0;
//	}
//	// insert current pid to proper memory space of the target gid
//	// todo: consider the situation that grid buffer is too small
//	uint cur_loc = atomicAdd(cur_grid,1);
//	if(cur_loc<bench->config->grid_capacity){
//		*(cur_grid+1+cur_loc) = pid;
//	}else{
//		atomicSub(cur_grid,1);
//	}
//}

__global__
void cleargrids_cuda(workbench *bench){
	int gid = blockIdx.x*blockDim.x+threadIdx.x;
	if(gid>=bench->num_grids){
		return;
	}
	*(bench->grids+(bench->config->grid_capacity+1)*gid) = 0;
}

__global__
void reset_bench_cuda(workbench *bench){
	bench->unit_lookup_counter = 0;
	bench->reaches_counter = 0;
}


//  partition with cuda
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
		curnode = bench->schema[curnode].children[loc];
		// is leaf
		if(bench->schema[curnode].isleaf){
			gid = bench->schema[curnode].node_id;
			break;
		}
	}

	// insert current pid to proper memory space of the target gid
	// todo: consider the situation that grid buffer is too small
	uint *cur_grid = bench->grids+(bench->config->grid_capacity+1)*gid;
	uint cur_loc = atomicAdd(cur_grid,1);
	if(cur_loc<bench->config->grid_capacity){
		*(cur_grid+1+cur_loc) = pid;
	}else{
		atomicSub(cur_grid,1);
	}
	uint glid = atomicAdd(&bench->unit_lookup_counter,1);
	bench->unit_lookup[glid].pid = pid;
	bench->unit_lookup[glid].gid = gid;
	bench->unit_lookup[glid].offset = 0;
	bench->unit_lookup[glid].inside = true;

	// need also check the upper and right nodes
	if(p->x+bench->config->x_buffer>bench->schema[curnode].mbr.high[0]||
	   p->y+bench->config->y_buffer>bench->schema[curnode].mbr.high[1]){
		uint stack_index = atomicAdd(&bench->stack_index[0],1);
		assert(stack_index<bench->stack_capacity);
		bench->lookup_stack[0][stack_index*2] = pid;
		bench->lookup_stack[0][stack_index*2+1] = 0;
	}
}

__global__
void pack_lookup_units_cuda(workbench *bench, uint inistial_size){
	int glid = blockIdx.x*blockDim.x+threadIdx.x;
	if(glid>=inistial_size){
		return;
	}

	uint grid_size = bench->grids[bench->unit_lookup[glid].gid
	                              *(bench->config->grid_capacity+1)];
	// the first batch already inserted during the partition and lookup steps
	uint offset = bench->config->zone_capacity;
	while(offset<grid_size){
		uint cu_index = atomicAdd(&bench->unit_lookup_counter, 1);
		assert(cu_index<bench->unit_lookup_capacity);
		bench->unit_lookup[cu_index] = bench->unit_lookup[glid];
		bench->unit_lookup[cu_index].offset = offset;
		offset += bench->config->zone_capacity;
	}
}


__device__
inline double distance(box *b,Point *p){

	double dx = max(abs(p->x-(b->low[0]+b->high[0])/2) - (b->high[0]-b->low[0])/2, 0.0);
	double dy = max(abs(p->y-(b->low[1]+b->high[1])/2) - (b->high[1]-b->low[1])/2, 0.0);
	dy = dy/degree_per_meter_latitude_cuda;
	dx = dx/degree_per_meter_longitude_cuda(p->y);

	return sqrt(dx * dx + dy * dy);
}

__device__
inline void print_box_point(box *b, Point *p){
	printf("POLYGON((%f %f, %f %f, %f %f, %f %f, %f %f))\nPOINT(%f %f)\n",
						b->low[0],b->low[1],
						b->high[0],b->low[1],
						b->high[0],b->high[1],
						b->low[0],b->high[1],
						b->low[0],b->low[1],
						p->x,p->y);
}

__device__
inline void print_box(box *b){
	printf("POLYGON((%f %f, %f %f, %f %f, %f %f, %f %f))\n",
						b->low[0],b->low[1],
						b->high[0],b->low[1],
						b->high[0],b->high[1],
						b->low[0],b->high[1],
						b->low[0],b->low[1]);
}

__device__
inline void print_point(Point *p){
	printf("Point(%f %f)\n",p->x,p->y);
}

__global__
void lookup_cuda(workbench *bench, uint stack_id, uint stack_size){

	int sid = blockIdx.x*blockDim.x+threadIdx.x;
	if(sid>=stack_size){
		return;
	}

	uint pid = bench->lookup_stack[stack_id][sid*2];
	uint curnode = bench->lookup_stack[stack_id][sid*2+1];
	Point *p = bench->points+pid;

	for(int i=0;i<4;i++){
		uint child_offset = bench->schema[curnode].children[i];
		double dist = distance(&bench->schema[child_offset].mbr, p);
		if(dist<=bench->config->reach_distance){
			if(bench->schema[child_offset].isleaf){
				if(dist>0){
					uint gid = bench->schema[child_offset].node_id;
					assert(gid<bench->num_grids);
					uint gl = atomicAdd(&bench->unit_lookup_counter,1);
					bench->unit_lookup[gl].pid = pid;
					bench->unit_lookup[gl].gid = gid;
					bench->unit_lookup[gl].offset = 0;
					bench->unit_lookup[gl].inside = false;
				}
			}else{
				uint idx = atomicAdd(&bench->stack_index[!stack_id],1);
				assert(idx<bench->stack_capacity);
				bench->lookup_stack[!stack_id][idx*2] = pid;
				bench->lookup_stack[!stack_id][idx*2+1] = child_offset;
			}
		}
	}

	// reset stack size for this round of checking
	if(sid==0){
		bench->stack_index[stack_id] = 0;
	}
}


__global__
void reachability_cuda(workbench *bench){

	// the objects in which grid need be processed
	int pairid = blockIdx.x*blockDim.x+threadIdx.x;
	if(pairid>=bench->unit_lookup_counter){
		return;
	}

	double max_dist = bench->config->reach_distance;
	uint pid = bench->unit_lookup[pairid].pid;
	uint gid = bench->unit_lookup[pairid].gid;
	uint offset = bench->unit_lookup[pairid].offset;
	uint size = *(bench->grids+(bench->config->grid_capacity+1)*gid)-offset;

	if(size>bench->config->zone_capacity){
		size = bench->config->zone_capacity;
	}
	//printf("%d\t%d\t%d\t%d\n",pid,gid,offset,size);

	const uint *cur_pids = bench->grids+(bench->config->grid_capacity+1)*gid+1+offset;
	for(uint i=0;i<size;i++){
		if(pid<cur_pids[i]||!bench->unit_lookup[pairid].inside){
			double dist = distance(bench->points[pid].x, bench->points[pid].y, bench->points[cur_pids[i]].x, bench->points[cur_pids[i]].y);
			if(dist<=max_dist){
				uint loc = atomicAdd(&bench->reaches_counter, 1);
				assert(loc<bench->reaches_capacity);
				bench->reaches[loc].pid1 = min(pid,cur_pids[i]);
				bench->reaches[loc].pid2 = max(cur_pids[i],pid);
			}
		}
	}
}

/*
 * in this phase, only update or append
 * */
__global__
void update_meetings_cuda(workbench *bench){

	int rid = blockIdx.x*blockDim.x+threadIdx.x;
	if(rid>=bench->reaches_counter){
		return;
	}
	uint pid1 = bench->reaches[rid].pid1;
	uint pid2 = bench->reaches[rid].pid2;
	uint bid = (pid1+pid2)%bench->config->num_meeting_buckets;
	meeting_unit *bucket = bench->meeting_buckets+bid*bench->meeting_bucket_capacity;
	bool updated = false;

	for(uint i=0;i<bench->meeting_buckets_counter_tmp[bid];i++){
		// a former meeting is encountered, update it
		if(bucket[i].pid1==pid1&&bucket[i].pid2==pid2){
			bucket[i].end = bench->cur_time;
			updated = true;
			break;
		}
	}

	// otherwise append it
	if(!updated){
		uint loc = atomicAdd(bench->meeting_buckets_counter+bid,1);
		assert(loc<bench->meeting_bucket_capacity);
		bucket[loc].pid1 = pid1;
		bucket[loc].pid2 = pid2;
		bucket[loc].start = bench->cur_time;
		bucket[loc].end = bench->cur_time;
	}
}

__global__
void compact_meetings_cuda(workbench *bench){
	int bid = blockIdx.x*blockDim.x+threadIdx.x;
	if(bid>=bench->config->num_meeting_buckets){
		return;
	}
	meeting_unit *bucket = bench->meeting_buckets+bid*bench->meeting_bucket_capacity;
	int front_idx = 0;
	int back_idx = bench->meeting_buckets_counter[bid]-1;
	uint meeting_idx = 0;
	int active_count = 0;
	for(;front_idx<=back_idx;front_idx++){
		// this meeting is over
		if(bucket[front_idx].end<bench->cur_time){
			// dump to valid list and copy one from the back end
			if(bucket[front_idx].end-bucket[front_idx].start>=bench->config->min_meet_time){
				meeting_idx = atomicAdd(&bench->meeting_counter,1);
				bench->meetings[meeting_idx] = bucket[front_idx];
			}
			for(;back_idx>front_idx;back_idx--){
				if(bucket[back_idx].end==bench->cur_time){
					break;
					// dump to valid list if needed or disregarded
				}else if(bucket[back_idx].end-bucket[back_idx].start>=bench->config->min_meet_time){
					meeting_idx = atomicAdd(&bench->meeting_counter,1);
					bench->meetings[meeting_idx] = bucket[back_idx];
				}
			}
			if(front_idx<back_idx){
				bucket[front_idx] = bucket[back_idx];
				active_count++;
				back_idx--;
			}
		}else{
			active_count++;
		}
	}
	bench->meeting_buckets_counter[bid] = active_count;
	bench->meeting_buckets_counter_tmp[bid] = active_count;
}

workbench *create_device_bench(workbench *bench, gpu_info *gpu){
	struct timeval start = get_cur_time();
	gpu->clear();
	// use h_bench as a container to copy in and out GPU
	workbench h_bench(bench);
	// space for the raw points data
	h_bench.points = (Point *)gpu->allocate(bench->config->num_objects*sizeof(Point));

	// space for the pids of all the grids
	h_bench.grids = (uint *)gpu->allocate(bench->num_grids*(bench->config->grid_capacity+1)*sizeof(uint));

	// space for the pid-zid pairs
	h_bench.unit_lookup = (checking_unit *)gpu->allocate(bench->unit_lookup_capacity*sizeof(checking_unit));
	// space for the QTtree schema
	h_bench.schema = (QTSchema *)gpu->allocate(bench->num_nodes*sizeof(QTSchema));
	// space for processing stack
	h_bench.lookup_stack[0] = (uint *)gpu->allocate(bench->stack_capacity*2*sizeof(uint));
	h_bench.lookup_stack[1] = (uint *)gpu->allocate(bench->stack_capacity*2*sizeof(uint));
	h_bench.reaches = (reach_unit *)gpu->allocate(bench->reaches_capacity*sizeof(reach_unit));

	h_bench.meeting_buckets = (meeting_unit *)gpu->allocate(bench->config->num_meeting_buckets*bench->meeting_bucket_capacity*sizeof(meeting_unit));
	h_bench.meeting_buckets_counter = (uint *)gpu->allocate(bench->config->num_meeting_buckets*sizeof(uint));
	h_bench.meeting_buckets_counter_tmp = (uint *)gpu->allocate(bench->config->num_meeting_buckets*sizeof(uint));
	h_bench.meetings = (meeting_unit *)gpu->allocate(bench->meeting_capacity*sizeof(meeting_unit));

	h_bench.config = (configuration *)gpu->allocate(sizeof(configuration));

	// space for the mapping of bench in GPU
	workbench *d_bench = (workbench *)gpu->allocate(sizeof(workbench));

	// the configuration and schema are fixed
	CUDA_SAFE_CALL(cudaMemcpy(h_bench.schema, bench->schema, bench->num_nodes*sizeof(QTSchema), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(h_bench.config, bench->config, sizeof(configuration), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_bench, &h_bench, sizeof(workbench), cudaMemcpyHostToDevice));

	logt("allocating space %d MB", start,gpu->size_allocated()/1024/1024);

	return d_bench;
}

/*
 *
 * check the reachability of objects in a list of partitions
 * ctx.data contains the list of
 *
 * */
void process_with_gpu(workbench *bench, workbench* d_bench, gpu_info *gpu){
	struct timeval start = get_cur_time();
	//gpu->print();
	assert(bench);
	assert(d_bench);
	assert(gpu);

	cudaSetDevice(gpu->device_id);
	// as temporary host workbench
	workbench h_bench(bench);
	CUDA_SAFE_CALL(cudaMemcpy(&h_bench, d_bench, sizeof(workbench), cudaMemcpyDeviceToHost));
	h_bench.cur_time = bench->cur_time;
	CUDA_SAFE_CALL(cudaMemcpy(d_bench, &h_bench, sizeof(workbench), cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL(cudaMemcpy(h_bench.points, bench->points, bench->config->num_objects*sizeof(Point), cudaMemcpyHostToDevice));
	logt("copying data", start);

	partition_cuda<<<bench->config->num_objects/1024+1,1024>>>(d_bench);
	check_execution();
	cudaDeviceSynchronize();
	CUDA_SAFE_CALL(cudaMemcpy(&h_bench, d_bench, sizeof(workbench), cudaMemcpyDeviceToHost));
	logt("partition data %d still need lookup", start,h_bench.stack_index[0]);

	uint stack_id = 0;
	while(h_bench.stack_index[stack_id]>0){
		lookup_cuda<<<h_bench.stack_index[stack_id]/1024+1,1024>>>(d_bench,stack_id,h_bench.stack_index[stack_id]);
		check_execution();
		cudaDeviceSynchronize();
		CUDA_SAFE_CALL(cudaMemcpy(&h_bench, d_bench, sizeof(workbench), cudaMemcpyDeviceToHost));
		stack_id = !stack_id;
	}
	logt("%d pid-grid pairs need be checked", start,h_bench.unit_lookup_counter);

	pack_lookup_units_cuda<<<h_bench.unit_lookup_counter/1024+1,1024>>>(d_bench,h_bench.unit_lookup_counter);
	check_execution();
	cudaDeviceSynchronize();
	CUDA_SAFE_CALL(cudaMemcpy(&h_bench, d_bench, sizeof(workbench), cudaMemcpyDeviceToHost));
	logt("%d pid-grid-offset tuples need be checked", start,h_bench.unit_lookup_counter);

	// compute the reachability of objects in each partitions
	reachability_cuda<<<h_bench.unit_lookup_counter/1024+1,1024>>>(d_bench);
	check_execution();
	cudaDeviceSynchronize();
	CUDA_SAFE_CALL(cudaMemcpy(&h_bench, d_bench, sizeof(workbench), cudaMemcpyDeviceToHost));
	logt("%d reaches are found", start,h_bench.reaches_counter);

	update_meetings_cuda<<<h_bench.reaches_counter/1024+1,1024>>>(d_bench);
	check_execution();
	cudaDeviceSynchronize();
	CUDA_SAFE_CALL(cudaMemcpy(&h_bench, d_bench, sizeof(workbench), cudaMemcpyDeviceToHost));
	logt("meeting buckets are updated", start);

	compact_meetings_cuda<<<bench->config->num_meeting_buckets/1024+1,1024>>>(d_bench);
	check_execution();
	cudaDeviceSynchronize();
	CUDA_SAFE_CALL(cudaMemcpy(&h_bench, d_bench, sizeof(workbench), cudaMemcpyDeviceToHost));
	logt("meeting buckets are compacted %d meetings are found", start, h_bench.meeting_counter);

	// todo for test only, should not copy out so much stuff
	if(bench->config->analyze){
		CUDA_SAFE_CALL(cudaMemcpy(bench->grids, h_bench.grids,
				bench->num_grids*(bench->config->grid_capacity+1)*sizeof(uint), cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(bench->meeting_buckets, h_bench.meeting_buckets,
				bench->config->num_meeting_buckets*bench->meeting_bucket_capacity*sizeof(meeting_unit), cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(bench->meeting_buckets_counter, h_bench.meeting_buckets_counter,
				bench->config->num_meeting_buckets*sizeof(uint), cudaMemcpyDeviceToHost));
	}
	if(h_bench.meeting_counter>0){
		bench->meeting_counter = h_bench.meeting_counter;
		CUDA_SAFE_CALL(cudaMemcpy(bench->meetings, h_bench.meetings,
				h_bench.meeting_counter*sizeof(meeting_unit), cudaMemcpyDeviceToHost));
	}
	logt("copy out", start);
	// clean the device bench for next round of checking
	cleargrids_cuda<<<bench->num_grids/1024+1,1024>>>(d_bench);
	//clear_meeting_buckets_cuda<<<bench->config->num_meeting_buckets/1024+1,1024>>>(d_bench);
	reset_bench_cuda<<<1,1>>>(d_bench);
	logt("clean", start);
}
