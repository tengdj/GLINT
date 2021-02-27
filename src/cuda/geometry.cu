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
//				assert(gid<bench->grids_counter);
//				uint offset = 0;
//				while(offset<bench->grids[gid*(bench->config->grid_capacity+1)]){
//					uint cu_index = atomicAdd(&bench->grid_check_counter, 1);
//					bench->grid_check[cu_index].pid = pid;
//					bench->grid_check[cu_index].gid = gid;
//					bench->grid_check[cu_index].offset = offset;
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


__global__
void cuda_cleargrids(workbench *bench){
	int gid = blockIdx.x*blockDim.x+threadIdx.x;
	if(gid>=bench->grids_stack_capacity){
		return;
	}
	bench->grid_counter[gid] = 0;
}

__global__
void cuda_reset_bench(workbench *bench){
	bench->grid_check_counter = 0;
	bench->reaches_counter = 0;
	bench->lookup_stack_index[0] = 0;
	bench->lookup_stack_index[1] = 0;
}


//  partition with cuda
__global__
void cuda_partition(workbench *bench){
	int pid = blockIdx.x*blockDim.x+threadIdx.x;
	if(pid>=bench->config->num_objects){
		return;
	}

	// search the tree to get in which grid
	uint curnode = 0;
	uint gid = 0;

	Point *p = bench->points+pid;
	while(true){
		int loc = (p->y>bench->schema[curnode].mid_y)*2 + (p->x>bench->schema[curnode].mid_x);
		curnode = bench->schema[curnode].children[loc];
		// is leaf
		if(bench->schema[curnode].type==LEAF){
			gid = bench->schema[curnode].grid_id;
			break;
		}
	}

	// insert current pid to proper memory space of the target gid
	// todo: consider the situation that grid buffer is too small
	uint *cur_grid = bench->grids+bench->grid_capacity*gid;
	uint cur_loc = atomicAdd(bench->grid_counter+gid,1);
	if(cur_loc<bench->grid_capacity){
		*(cur_grid+cur_loc) = pid;
	}
	uint glid = atomicAdd(&bench->grid_check_counter,1);
	bench->grid_check[glid].pid = pid;
	bench->grid_check[glid].gid = gid;
	bench->grid_check[glid].offset = 0;
	bench->grid_check[glid].inside = true;

	// need also check the upper and right nodes
	if(p->x+bench->config->x_buffer>bench->schema[curnode].mbr.high[0]||
	   p->y+bench->config->y_buffer>bench->schema[curnode].mbr.high[1]){
		uint stack_index = atomicAdd(&bench->lookup_stack_index[0],1);
		assert(stack_index<bench->lookup_stack_capacity);
		bench->lookup_stack[0][stack_index*2] = pid;
		bench->lookup_stack[0][stack_index*2+1] = 0;
	}
}

__global__
void cuda_pack_lookup_units(workbench *bench, uint inistial_size){
	int glid = blockIdx.x*blockDim.x+threadIdx.x;
	if(glid>=inistial_size){
		return;
	}

	uint grid_size = min(bench->grid_counter[bench->grid_check[glid].gid],bench->grid_capacity);
	// the first batch already inserted during the partition and lookup steps
	uint offset = bench->config->zone_capacity;
	while(offset<grid_size){
		uint cu_index = atomicAdd(&bench->grid_check_counter, 1);
//		if(cu_index>=bench->grid_check_capacity){
//			printf("%d %d %d\n",bench->grid_counter[bench->grid_check[glid].gid],cu_index,bench->grid_check_capacity);
//		}
		assert(cu_index<bench->grid_check_capacity);
		bench->grid_check[cu_index] = bench->grid_check[glid];
		bench->grid_check[cu_index].offset = offset;
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
void cuda_lookup(workbench *bench, uint stack_id, uint stack_size){

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
			if(bench->schema[child_offset].type==LEAF){
				if(p->y<bench->schema[child_offset].mbr.low[1]||
				   (p->y<bench->schema[child_offset].mbr.high[1]
					&& p->x<bench->schema[child_offset].mbr.low[0])){
					uint gid = bench->schema[child_offset].grid_id;
					assert(gid<bench->grids_stack_capacity);
					uint gl = atomicAdd(&bench->grid_check_counter,1);
					bench->grid_check[gl].pid = pid;
					bench->grid_check[gl].gid = gid;
					bench->grid_check[gl].offset = 0;
					bench->grid_check[gl].inside = false;
				}
			}else{
				uint idx = atomicAdd(&bench->lookup_stack_index[!stack_id],1);
				assert(idx<bench->lookup_stack_capacity);
				bench->lookup_stack[!stack_id][idx*2] = pid;
				bench->lookup_stack[!stack_id][idx*2+1] = child_offset;
			}
		}
	}

	// reset stack size for this round of checking
	if(sid==0){
		bench->lookup_stack_index[stack_id] = 0;
	}
}


__global__
void cuda_reachability(workbench *bench){

	// the objects in which grid need be processed
	int loc = threadIdx.y;
	int pairid = blockIdx.x*blockDim.x+threadIdx.x;
	if(pairid>=bench->grid_check_counter){
		return;
	}

	uint gid = bench->grid_check[pairid].gid;
	uint offset = bench->grid_check[pairid].offset;
	uint size = min(bench->grid_counter[gid],bench->grid_capacity)-offset;
	if(size>bench->config->zone_capacity){
		size = bench->config->zone_capacity;
	}
	if(loc>=size){
		return;
	}

	uint pid = bench->grid_check[pairid].pid;
	uint target_pid = *(bench->grids+bench->grid_capacity*gid+offset+loc);
	if(!bench->grid_check[pairid].inside||pid<target_pid){
		double dist = distance(bench->points[pid].x, bench->points[pid].y, bench->points[target_pid].x, bench->points[target_pid].y);
		if(dist<=bench->config->reach_distance){
			uint loc = atomicAdd(&bench->reaches_counter, 1);
			assert(loc<bench->reaches_capacity);
			bench->reaches[loc].pid1 = min(pid,target_pid);
			bench->reaches[loc].pid2 = max(target_pid,pid);
		}
	}

}

/*
 * in this phase, only update or append
 * */
__global__
void cuda_update_meetings(workbench *bench){

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
void cuda_compact_meetings(workbench *bench){
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


__device__
inline void merge_node(workbench *bench, uint cur_node){
	assert(bench->schema[cur_node].type==BRANCH);
	//printf("merge\n");

	//reclaim the children
	uint gid = 0;
	for(int i=0;i<4;i++){
		uint child_offset = bench->schema[cur_node].children[i];
		assert(bench->schema[child_offset].type==LEAF);
		//bench->schema[child_offset].mbr.print();
		// push the bench->schema and grid spaces to stack for reuse

		bench->grid_counter[bench->schema[child_offset].grid_id] = 0;
		if(i<3){
			// push to stack
			uint idx = atomicSub(&bench->grids_stack_index,1)-1;
			bench->grids_stack[idx] = bench->schema[child_offset].grid_id;
		}else{
			// reused by curnode
			gid = bench->schema[child_offset].grid_id;
		}
		bench->schema[child_offset].type = INVALID;
		uint idx = atomicSub(&bench->schema_stack_index,1)-1;
		bench->schema_stack[idx] = child_offset;
	}
	bench->schema[cur_node].type = LEAF;
	// reuse the grid of one of its child
	bench->schema[cur_node].grid_id = gid;
}


__device__
inline void split_node(workbench *bench, uint cur_node){
	assert(bench->schema[cur_node].type==LEAF);
	//printf("split\n");
	//schema[cur_node].mbr.print();
	bench->schema[cur_node].type = BRANCH;
	// reuse by one of its child
	uint gid = bench->schema[cur_node].grid_id;

	double xhalf = bench->schema[cur_node].mid_x-bench->schema[cur_node].mbr.low[0];
	double yhalf = bench->schema[cur_node].mid_y-bench->schema[cur_node].mbr.low[1];

	for(int i=0;i<4;i++){
		// pop space for schema and grid
		uint idx = atomicAdd(&bench->schema_stack_index, 1);
		assert(idx<bench->schema_stack_capacity);
		uint child = bench->schema_stack[idx];
		bench->schema[cur_node].children[i] = child;

		if(i>0){
			idx = atomicAdd(&bench->grids_stack_index,1);
			assert(idx<bench->grids_stack_capacity);
			gid = bench->grids_stack[idx];
		}
		bench->schema[child].grid_id = gid;
		bench->grid_counter[gid] = 0;
		bench->schema[child].level = bench->schema[cur_node].level+1;
		bench->schema[child].type = LEAF;
		bench->schema[child].overflow_count = 0;
		bench->schema[child].underflow_count = 0;

		bench->schema[child].mbr.low[0] = bench->schema[cur_node].mbr.low[0]+(i%2==1)*xhalf;
		bench->schema[child].mbr.low[1] = bench->schema[cur_node].mbr.low[1]+(i/2==1)*yhalf;
		bench->schema[child].mbr.high[0] = bench->schema[cur_node].mid_x+(i%2==1)*xhalf;
		bench->schema[child].mbr.high[1] = bench->schema[cur_node].mid_y+(i/2==1)*yhalf;
		bench->schema[child].mid_x = (bench->schema[child].mbr.low[0]+bench->schema[child].mbr.high[0])/2;
		bench->schema[child].mid_y = (bench->schema[child].mbr.low[1]+bench->schema[child].mbr.high[1])/2;
		//schema[child].mbr.print();
	}
}

__global__
void cuda_update_schema_conduct(workbench *bench, uint size){
	uint sidx = blockIdx.x*blockDim.x+threadIdx.x;
	if(sidx>=size){
		return;
	}
	uint curnode = bench->lookup_stack[0][sidx];
	//printf("%d\n",curnode);
	if(bench->schema[curnode].type==LEAF){
		printf("split: %d\n",curnode);
		split_node(bench,curnode);
	}else{
		printf("merge: %d\n",curnode);
		merge_node(bench,curnode);
	}
}

__global__
void cuda_update_schema_collect(workbench *bench){
	uint curnode = blockIdx.x*blockDim.x+threadIdx.x;
	if(curnode>=bench->schema_stack_capacity){
		return;
	}
	if(bench->schema[curnode].type==LEAF){
		if(bench->grid_counter[bench->schema[curnode].grid_id]>bench->config->grid_capacity){
			// this node is overflowed a continuous number of times, split it
			if(++bench->schema[curnode].overflow_count>=bench->config->schema_update_delay){
				uint sidx = atomicAdd(&bench->lookup_stack_index[0],1);
				bench->lookup_stack[0][sidx] = curnode;
				bench->schema[curnode].overflow_count = 0;
			}
		}else{
			bench->schema[curnode].overflow_count = 0;
		}
	}else if(bench->schema[curnode].type==BRANCH){
		int leafchild = 0;
		int ncounter = 0;
		for(int i=0;i<4;i++){
			uint child_node = bench->schema[curnode].children[i];
			if(bench->schema[child_node].type==LEAF){
				leafchild++;
				ncounter += bench->grid_counter[bench->schema[child_node].grid_id];
			}
		}
		// this one need update
		if(leafchild==4&&ncounter<bench->config->grid_capacity){
			// the children of this node need be deallocated
			if(++bench->schema[curnode].underflow_count>=bench->config->schema_update_delay){
				//printf("%d\n",curnode);
				uint sidx = atomicAdd(&bench->lookup_stack_index[0],1);
				bench->lookup_stack[0][sidx] = curnode;
				bench->schema[curnode].underflow_count = 0;
			}
		}else{
			bench->schema[curnode].underflow_count = 0;
		}
	}
}

workbench *create_device_bench(workbench *bench, gpu_info *gpu){
	struct timeval start = get_cur_time();
	gpu->clear();
	// use h_bench as a container to copy in and out GPU
	workbench h_bench(bench);
	// space for the raw points data
	h_bench.points = (Point *)gpu->allocate(bench->config->num_objects*sizeof(Point));

	// space for the pids of all the grids
	h_bench.grids = (uint *)gpu->allocate(bench->grids_stack_capacity*bench->grid_capacity*sizeof(uint));
	h_bench.grid_counter = (uint *)gpu->allocate(bench->grids_stack_capacity*sizeof(uint));
	h_bench.grids_stack = (uint *)gpu->allocate(bench->grids_stack_capacity*sizeof(uint));

	// space for the pid-zid pairs
	h_bench.grid_check = (checking_unit *)gpu->allocate(bench->grid_check_capacity*sizeof(checking_unit));

	// space for the QTtree schema
	h_bench.schema = (QTSchema *)gpu->allocate(bench->schema_stack_capacity*sizeof(QTSchema));
	h_bench.schema_stack = (uint *)gpu->allocate(bench->schema_stack_capacity*sizeof(uint));

	// space for processing stack
	h_bench.lookup_stack[0] = (uint *)gpu->allocate(bench->lookup_stack_capacity*2*sizeof(uint));
	h_bench.lookup_stack[1] = (uint *)gpu->allocate(bench->lookup_stack_capacity*2*sizeof(uint));
	h_bench.reaches = (reach_unit *)gpu->allocate(bench->reaches_capacity*sizeof(reach_unit));

	h_bench.meeting_buckets = (meeting_unit *)gpu->allocate(bench->config->num_meeting_buckets*bench->meeting_bucket_capacity*sizeof(meeting_unit));
	h_bench.meeting_buckets_counter = (uint *)gpu->allocate(bench->config->num_meeting_buckets*sizeof(uint));
	h_bench.meeting_buckets_counter_tmp = (uint *)gpu->allocate(bench->config->num_meeting_buckets*sizeof(uint));
	h_bench.meetings = (meeting_unit *)gpu->allocate(bench->meeting_capacity*sizeof(meeting_unit));

	h_bench.config = (configuration *)gpu->allocate(sizeof(configuration));

	// space for the mapping of bench in GPU
	workbench *d_bench = (workbench *)gpu->allocate(sizeof(workbench));

	// the configuration and schema are fixed
	CUDA_SAFE_CALL(cudaMemcpy(h_bench.schema, bench->schema, bench->schema_stack_capacity*sizeof(QTSchema), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(h_bench.config, bench->config, sizeof(configuration), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_bench, &h_bench, sizeof(workbench), cudaMemcpyHostToDevice));

	logt("GPU allocating space %ld MB", start,gpu->size_allocated()/1024/1024);

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

	cuda_partition<<<bench->config->num_objects/1024+1,1024>>>(d_bench);
	check_execution();
	cudaDeviceSynchronize();
	CUDA_SAFE_CALL(cudaMemcpy(&h_bench, d_bench, sizeof(workbench), cudaMemcpyDeviceToHost));
	logt("partition data %d still need lookup", start,h_bench.lookup_stack_index[0]);

	uint stack_id = 0;
	while(h_bench.lookup_stack_index[stack_id]>0){
		cuda_lookup<<<h_bench.lookup_stack_index[stack_id]/1024+1,1024>>>(d_bench,stack_id,h_bench.lookup_stack_index[stack_id]);
		check_execution();
		cudaDeviceSynchronize();
		CUDA_SAFE_CALL(cudaMemcpy(&h_bench, d_bench, sizeof(workbench), cudaMemcpyDeviceToHost));
		stack_id = !stack_id;
	}
	logt("%d pid-grid pairs need be checked", start,h_bench.grid_check_counter);

	cuda_pack_lookup_units<<<h_bench.grid_check_counter/1024+1,1024>>>(d_bench,h_bench.grid_check_counter);
	check_execution();
	cudaDeviceSynchronize();
	CUDA_SAFE_CALL(cudaMemcpy(&h_bench, d_bench, sizeof(workbench), cudaMemcpyDeviceToHost));
	logt("%d pid-grid-offset tuples need be checked", start,h_bench.grid_check_counter);


	// compute the reachability of objects in each partitions
	uint thread_y = bench->config->zone_capacity;
	uint thread_x = 1024/thread_y;
	dim3 block(thread_x, thread_y);

	cuda_reachability<<<h_bench.grid_check_counter/thread_x+1,block>>>(d_bench);
	check_execution();
	cudaDeviceSynchronize();
	CUDA_SAFE_CALL(cudaMemcpy(&h_bench, d_bench, sizeof(workbench), cudaMemcpyDeviceToHost));
	logt("%d reaches are found %d", start,h_bench.reaches_counter,h_bench.test_counter);

	cuda_update_meetings<<<h_bench.reaches_counter/1024+1,1024>>>(d_bench);
	check_execution();
	cudaDeviceSynchronize();
	CUDA_SAFE_CALL(cudaMemcpy(&h_bench, d_bench, sizeof(workbench), cudaMemcpyDeviceToHost));
	logt("meeting buckets are updated", start);

	cuda_compact_meetings<<<bench->config->num_meeting_buckets/1024+1,1024>>>(d_bench);
	check_execution();
	cudaDeviceSynchronize();
	CUDA_SAFE_CALL(cudaMemcpy(&h_bench, d_bench, sizeof(workbench), cudaMemcpyDeviceToHost));
	logt("meeting buckets are compacted %d meetings are found", start, h_bench.meeting_counter);

	// todo for test only, should not copy out so much stuff
	if(bench->config->analyze){
		CUDA_SAFE_CALL(cudaMemcpy(bench->grids, h_bench.grids,
				bench->grids_stack_capacity*bench->grid_capacity*sizeof(uint), cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(bench->grid_counter, h_bench.grid_counter,
				bench->grids_stack_capacity*sizeof(uint), cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(bench->schema, h_bench.schema,
				bench->schema_stack_capacity*sizeof(QTSchema), cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(bench->meeting_buckets, h_bench.meeting_buckets,
				bench->config->num_meeting_buckets*bench->meeting_bucket_capacity*sizeof(meeting_unit), cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(bench->meeting_buckets_counter, h_bench.meeting_buckets_counter,
				bench->config->num_meeting_buckets*sizeof(uint), cudaMemcpyDeviceToHost));
		if(h_bench.meeting_counter>0){
			bench->meeting_counter = h_bench.meeting_counter;
			CUDA_SAFE_CALL(cudaMemcpy(bench->meetings, h_bench.meetings,
					h_bench.meeting_counter*sizeof(meeting_unit), cudaMemcpyDeviceToHost));
		}
		bench->schema_stack_index = h_bench.schema_stack_index;
		bench->grids_stack_index = h_bench.grids_stack_index;
		logt("copy out", start);
	}

	if(bench->config->dynamic_schema){
		// update the schema for future processing
		cuda_update_schema_collect<<<bench->schema_stack_capacity,1024>>>(d_bench);
		check_execution();
		cudaDeviceSynchronize();
		CUDA_SAFE_CALL(cudaMemcpy(&h_bench, d_bench, sizeof(workbench), cudaMemcpyDeviceToHost));
		if(h_bench.lookup_stack_index[0]>0){
			cuda_update_schema_conduct<<<h_bench.lookup_stack_index[0],1024>>>(d_bench, h_bench.lookup_stack_index[0]);
			check_execution();
			cudaDeviceSynchronize();
		}
		logt("schema update", start);
	}

	// clean the device bench for next round of checking
	cuda_cleargrids<<<bench->grids_stack_capacity/1024+1,1024>>>(d_bench);
	cuda_reset_bench<<<1,1>>>(d_bench);
	logt("clean", start);
}
