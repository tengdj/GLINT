#include <cuda.h>
#include "mygpu.h"
#include "cuda_util.h"
#include "../geometry/geometry.h"
#include "../util/query_context.h"
#include "../tracing/partitioner.h"
#include "../tracing/workbench.h"


/*
 *
 * some utility functions
 *
 * */

__device__
inline double height(box *b){
	return (b->high[1]-b->low[1])/degree_per_meter_latitude_cuda;
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
inline double contain(box *b, Point *p){
	return p->x>=b->low[0]&&
		   p->x<=b->high[0]&&
		   p->y>=b->low[1]&&
		   p->y<=b->high[1];
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
	bench->global_stack_index[0] = 0;
	bench->global_stack_index[1] = 0;
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
	uint last_valid = 0;
	while(true){
		int loc = (p->y>bench->schema[curnode].mid_y)*2 + (p->x>bench->schema[curnode].mid_x);
		curnode = bench->schema[curnode].children[loc];

		// not near the right and top border
		if(p->x+bench->config->x_buffer<bench->schema[curnode].mbr.high[0]&&
		   p->y+bench->config->y_buffer<bench->schema[curnode].mbr.high[1]){
			last_valid = curnode;
		}

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

	if(last_valid!=curnode){
		uint stack_index = atomicAdd(&bench->global_stack_index[0],1);
		assert(stack_index<bench->global_stack_capacity);
		bench->global_stack[0][stack_index*2] = pid;
		bench->global_stack[0][stack_index*2+1] = last_valid;
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

__global__
void cuda_lookup(workbench *bench, uint stack_id, uint stack_size){

	int sid = blockIdx.x*blockDim.x+threadIdx.x;
	if(sid>=stack_size){
		return;
	}

	uint pid = bench->global_stack[stack_id][sid*2];
	uint curnode = bench->global_stack[stack_id][sid*2+1];
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
					//assert(gid<bench->grids_stack_capacity);
					uint gl = atomicAdd(&bench->grid_check_counter,1);
					bench->grid_check[gl].pid = pid;
					bench->grid_check[gl].gid = gid;
					bench->grid_check[gl].offset = 0;
					bench->grid_check[gl].inside = false;
				}
			}else{
				uint idx = atomicAdd(&bench->global_stack_index[!stack_id],1);
				//assert(idx<bench->global_stack_capacity);
				bench->global_stack[!stack_id][idx*2] = pid;
				bench->global_stack[!stack_id][idx*2+1] = child_offset;
			}
		}
	}

	// reset stack size for this round of checking
	if(sid==0){
		bench->global_stack_index[stack_id] = 0;
	}
}

#define PER_STACK_SIZE 30

__global__
void cuda_lookup_recursive(workbench *bench, int start_pid){

	int cur_pid = blockIdx.x*blockDim.x+threadIdx.x;
	int pid = cur_pid + start_pid;
	if(pid>=bench->config->num_objects){
		return;
	}

	uint *cur_stack = bench->global_stack[0]+cur_pid*PER_STACK_SIZE;
	assert(cur_pid*PER_STACK_SIZE<2*bench->global_stack_capacity);

	uint stack_index = 0;
	Point *p = bench->points+pid;
	cur_stack[stack_index++] = 0;
	while(stack_index>0){
		uint curnode = cur_stack[--stack_index];
		for(int i=0;i<4;i++){
			uint child_offset = bench->schema[curnode].children[i];
			double dist = distance(&bench->schema[child_offset].mbr, p);
			if(dist<=bench->config->reach_distance){
				if(bench->schema[child_offset].type==LEAF){
					uint gid = bench->schema[child_offset].grid_id;
					assert(gid<bench->grids_stack_capacity);
					if(contain(&bench->schema[child_offset].mbr,p)){
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
					}else if(p->y<bench->schema[child_offset].mbr.low[1]||
					   (p->y<bench->schema[child_offset].mbr.high[1]
						&& p->x<bench->schema[child_offset].mbr.low[0])){
						uint glid = atomicAdd(&bench->grid_check_counter,1);
						bench->grid_check[glid].pid = pid;
						bench->grid_check[glid].gid = gid;
						bench->grid_check[glid].offset = 0;
						bench->grid_check[glid].inside = false;
					}
				}else{
					assert(stack_index<PER_STACK_SIZE);
					cur_stack[stack_index++] = child_offset;
				}
			}
		}
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
			uint bid = (pid+target_pid)%bench->config->num_meeting_buckets;
			uint loc = atomicAdd(bench->meeting_buckets_counter[bench->current_bucket]+bid,1);

			// todo handling overflow
			if(loc<bench->meeting_bucket_capacity){
				meeting_unit *bucket = bench->meeting_buckets[bench->current_bucket]+bid*bench->meeting_bucket_capacity;
				bucket[loc].pid1 = min(pid,target_pid);
				bucket[loc].pid2 = max(target_pid,pid);
				bucket[loc].start = bench->cur_time;
			}
		}
	}
}

/*
 * update the first meet
 * */
__global__
void cuda_update_meetings(workbench *bench){

	int bid = blockIdx.x*blockDim.x+threadIdx.x;
	if(bid>=bench->config->num_meeting_buckets){
		return;
	}

	meeting_unit *bucket_new = bench->meeting_buckets[bench->current_bucket]+bid*bench->meeting_bucket_capacity;
	meeting_unit *bucket_old = bench->meeting_buckets[!bench->current_bucket]+bid*bench->meeting_bucket_capacity;

	uint size_new = bench->meeting_buckets_counter[bench->current_bucket][bid];
	uint size_old = bench->meeting_buckets_counter[!bench->current_bucket][bid];

	for(uint i=0;i<size_old&&i<bench->meeting_bucket_capacity;i++){
		bool updated = false;
		for(uint j=0;j<size_new&&j<bench->meeting_bucket_capacity;j++){
			if(bucket_new[i].pid1==bucket_old[j].pid1&&
			   bucket_new[i].pid2==bucket_old[j].pid2){
				bucket_new[i].start = bucket_old[i].start;
				updated = true;
				break;
			}
		}
		// the old meeting is over
		if(!updated&&
			bench->cur_time - bucket_old[i].start>=bench->config->min_meet_time){
			uint meeting_idx = atomicAdd(&bench->meeting_counter,1);
			assert(meeting_idx<bench->meeting_capacity);
			bench->meetings[meeting_idx] = bucket_old[i];
		}
	}
	// reset the old buckets for next batch of processing
	bench->meeting_buckets_counter[!bench->current_bucket][bid] = 0;
}

__global__
void cuda_update_schema_split(workbench *bench, uint size){
	uint sidx = blockIdx.x*blockDim.x+threadIdx.x;
	if(sidx>=size){
		return;
	}
	uint curnode = bench->global_stack[0][sidx];
	//printf("split: %d\n",curnode);
	//schema[curnode].mbr.print();
	bench->schema[curnode].type = BRANCH;
	// reuse by one of its child
	uint gid = bench->schema[curnode].grid_id;

	double xhalf = bench->schema[curnode].mid_x-bench->schema[curnode].mbr.low[0];
	double yhalf = bench->schema[curnode].mid_y-bench->schema[curnode].mbr.low[1];

	for(int i=0;i<4;i++){
		// pop space for schema and grid
		uint idx = atomicAdd(&bench->schema_stack_index, 1);
		assert(idx<bench->schema_stack_capacity);
		uint child = bench->schema_stack[idx];
		//printf("sidx: %d %d\n",idx,child);
		bench->schema[curnode].children[i] = child;

		if(i>0){
			idx = atomicAdd(&bench->grids_stack_index,1);
			assert(idx<bench->grids_stack_capacity);
			gid = bench->grids_stack[idx];
			//printf("gidx: %d %d\n",idx,gid);
		}
		bench->schema[child].grid_id = gid;
		bench->grid_counter[gid] = 0;
		bench->schema[child].level = bench->schema[curnode].level+1;
		bench->schema[child].type = LEAF;
		bench->schema[child].overflow_count = 0;
		bench->schema[child].underflow_count = 0;

		bench->schema[child].mbr.low[0] = bench->schema[curnode].mbr.low[0]+(i%2==1)*xhalf;
		bench->schema[child].mbr.low[1] = bench->schema[curnode].mbr.low[1]+(i/2==1)*yhalf;
		bench->schema[child].mbr.high[0] = bench->schema[curnode].mid_x+(i%2==1)*xhalf;
		bench->schema[child].mbr.high[1] = bench->schema[curnode].mid_y+(i/2==1)*yhalf;
		bench->schema[child].mid_x = (bench->schema[child].mbr.low[0]+bench->schema[child].mbr.high[0])/2;
		bench->schema[child].mid_y = (bench->schema[child].mbr.low[1]+bench->schema[child].mbr.high[1])/2;
	}

}
__global__
void cuda_update_schema_merge(workbench *bench, uint size){
	uint sidx = blockIdx.x*blockDim.x+threadIdx.x;
	if(sidx>=size){
		return;
	}
	uint curnode = bench->global_stack[1][sidx];
	//reclaim the children
	uint gid = 0;
	for(int i=0;i<4;i++){
		uint child_offset = bench->schema[curnode].children[i];
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
	bench->schema[curnode].type = LEAF;
	// reuse the grid of one of its child
	bench->schema[curnode].grid_id = gid;
}

__global__
void cuda_update_schema_collect(workbench *bench){
	uint curnode = blockIdx.x*blockDim.x+threadIdx.x;
	if(curnode>=bench->schema_stack_capacity){
		return;
	}
	if(bench->schema[curnode].type==LEAF){
		if(height(&bench->schema[curnode].mbr)>2*bench->config->reach_distance&&
				bench->grid_counter[bench->schema[curnode].grid_id]>bench->config->grid_capacity){
			// this node is overflowed a continuous number of times, split it
			if(++bench->schema[curnode].overflow_count>=bench->config->schema_update_delay){
				uint sidx = atomicAdd(&bench->global_stack_index[0],1);
				bench->global_stack[0][sidx] = curnode;
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
			// this node need be merged
			if(++bench->schema[curnode].underflow_count>=bench->config->schema_update_delay){
				//printf("%d\n",curnode);
				uint sidx = atomicAdd(&bench->global_stack_index[1],1);
				bench->global_stack[1][sidx] = curnode;
				bench->schema[curnode].underflow_count = 0;
			}
		}else{
			bench->schema[curnode].underflow_count = 0;
		}
	}
}


__global__
void cuda_init_schema_stack(workbench *bench){
	uint curnode = blockIdx.x*blockDim.x+threadIdx.x;
	if(curnode>=bench->schema_stack_capacity){
		return;
	}
	bench->schema_stack[curnode] = curnode;
}
__global__
void cuda_init_grids_stack(workbench *bench){
	uint curnode = blockIdx.x*blockDim.x+threadIdx.x;
	if(curnode>=bench->grids_stack_capacity){
		return;
	}
	bench->grids_stack[curnode] = curnode;
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
	h_bench.global_stack[0] = (uint *)gpu->allocate(bench->global_stack_capacity*2*sizeof(uint));
	h_bench.global_stack[1] = (uint *)gpu->allocate(bench->global_stack_capacity*2*sizeof(uint));

	h_bench.meeting_buckets[0] = (meeting_unit *)gpu->allocate(bench->config->num_meeting_buckets*bench->meeting_bucket_capacity*sizeof(meeting_unit));
	h_bench.meeting_buckets[1] = (meeting_unit *)gpu->allocate(bench->config->num_meeting_buckets*bench->meeting_bucket_capacity*sizeof(meeting_unit));
	h_bench.meeting_buckets_counter[0] = (uint *)gpu->allocate(bench->config->num_meeting_buckets*sizeof(uint));
	h_bench.meeting_buckets_counter[1] = (uint *)gpu->allocate(bench->config->num_meeting_buckets*sizeof(uint));
	h_bench.meetings = (meeting_unit *)gpu->allocate(bench->meeting_capacity*sizeof(meeting_unit));

	h_bench.config = (configuration *)gpu->allocate(sizeof(configuration));

	// space for the mapping of bench in GPU
	workbench *d_bench = (workbench *)gpu->allocate(sizeof(workbench));

	// the configuration and schema are fixed
	CUDA_SAFE_CALL(cudaMemcpy(h_bench.schema, bench->schema, bench->schema_stack_capacity*sizeof(QTSchema), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(h_bench.config, bench->config, sizeof(configuration), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_bench, &h_bench, sizeof(workbench), cudaMemcpyHostToDevice));

	cuda_init_grids_stack<<<bench->grids_stack_capacity/1024, 1024>>>(d_bench);
	cuda_init_schema_stack<<<bench->schema_stack_capacity/1024, 1024>>>(d_bench);



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

	// setup the current time and points for this round
	workbench h_bench(bench);
	CUDA_SAFE_CALL(cudaMemcpy(&h_bench, d_bench, sizeof(workbench), cudaMemcpyDeviceToHost));
	h_bench.cur_time = bench->cur_time;
	h_bench.current_bucket = bench->current_bucket;
	CUDA_SAFE_CALL(cudaMemcpy(d_bench, &h_bench, sizeof(workbench), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(h_bench.points, bench->points, bench->config->num_objects*sizeof(Point), cudaMemcpyHostToDevice));
	logt("copy in data", start);


	// do the tree lookup
	if(bench->config->recursive_lookup){
		uint batch_size = 1024000;
		for(int i=0;i<bench->config->num_objects;i+=batch_size){
			cuda_lookup_recursive<<<min(batch_size,bench->config->num_objects-i)/1024+1,1024>>>(d_bench, i);
			check_execution();
			cudaDeviceSynchronize();
		}
		CUDA_SAFE_CALL(cudaMemcpy(&h_bench, d_bench, sizeof(workbench), cudaMemcpyDeviceToHost));
		logt("partition data %d checkings", start,h_bench.grid_check_counter);
	}else{
		// do the partition
		cuda_partition<<<bench->config->num_objects/1024+1,1024>>>(d_bench);
		check_execution();
		cudaDeviceSynchronize();
		CUDA_SAFE_CALL(cudaMemcpy(&h_bench, d_bench, sizeof(workbench), cudaMemcpyDeviceToHost));
		logt("partition data %d still need lookup", start,h_bench.global_stack_index[0]);

		uint stack_id = 0;
		while(h_bench.global_stack_index[stack_id]>0){
			cuda_lookup<<<h_bench.global_stack_index[stack_id]/1024+1,1024>>>(d_bench,stack_id,h_bench.global_stack_index[stack_id]);
			check_execution();
			cudaDeviceSynchronize();
			CUDA_SAFE_CALL(cudaMemcpy(&h_bench, d_bench, sizeof(workbench), cudaMemcpyDeviceToHost));
			stack_id = !stack_id;
		}
		logt("%d pid-grid pairs need be checked", start,h_bench.grid_check_counter);
	}

	// do the pack
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
	logt("reaches computation", start);

	// update the meeting hash table
	uint origin_num_meeting = h_bench.meeting_counter;
	cuda_update_meetings<<<bench->config->num_meeting_buckets/1024+1,1024>>>(d_bench);
	check_execution();
	cudaDeviceSynchronize();
	CUDA_SAFE_CALL(cudaMemcpy(&h_bench, d_bench, sizeof(workbench), cudaMemcpyDeviceToHost));
	logt("meeting buckets update %d new meetings found", start, h_bench.meeting_counter-origin_num_meeting);

	// todo do the data analyzes, for test only, should not copy out so much stuff
	do{
		if(bench->config->analyze_grid||bench->config->analyze_reach){
			CUDA_SAFE_CALL(cudaMemcpy(bench->grids, h_bench.grids,
					bench->grids_stack_capacity*bench->grid_capacity*sizeof(uint), cudaMemcpyDeviceToHost));
			CUDA_SAFE_CALL(cudaMemcpy(bench->grid_counter, h_bench.grid_counter,
					bench->grids_stack_capacity*sizeof(uint), cudaMemcpyDeviceToHost));
			CUDA_SAFE_CALL(cudaMemcpy(bench->schema, h_bench.schema,
					bench->schema_stack_capacity*sizeof(QTSchema), cudaMemcpyDeviceToHost));
			bench->schema_stack_index = h_bench.schema_stack_index;
			bench->grids_stack_index = h_bench.grids_stack_index;
			logt("copy out grid and schema data", start);
		}
		if(bench->config->analyze_meeting||bench->config->analyze_reach){
			CUDA_SAFE_CALL(cudaMemcpy(bench->meeting_buckets[bench->current_bucket], h_bench.meeting_buckets[bench->current_bucket],
					bench->config->num_meeting_buckets*bench->meeting_bucket_capacity*sizeof(meeting_unit), cudaMemcpyDeviceToHost));
			CUDA_SAFE_CALL(cudaMemcpy(bench->meeting_buckets_counter[bench->current_bucket], h_bench.meeting_buckets_counter[bench->current_bucket],
					bench->config->num_meeting_buckets*sizeof(uint), cudaMemcpyDeviceToHost));
			if(h_bench.meeting_counter>0){
				bench->meeting_counter = h_bench.meeting_counter;
				CUDA_SAFE_CALL(cudaMemcpy(bench->meetings, h_bench.meetings, h_bench.meeting_counter*sizeof(meeting_unit), cudaMemcpyDeviceToHost));
			}
			logt("copy out meeting data", start);
		}
	}while(false);

	// do the schema update
	if(bench->config->dynamic_schema){
		// update the schema for future processing
		cuda_update_schema_collect<<<bench->schema_stack_capacity,1024>>>(d_bench);
		check_execution();
		cudaDeviceSynchronize();
		CUDA_SAFE_CALL(cudaMemcpy(&h_bench, d_bench, sizeof(workbench), cudaMemcpyDeviceToHost));
		if(h_bench.global_stack_index[0]>0){
			cuda_update_schema_split<<<h_bench.global_stack_index[0],1024>>>(d_bench, h_bench.global_stack_index[0]);
			check_execution();
			cudaDeviceSynchronize();
		}
		if(h_bench.global_stack_index[1]>0){
			cuda_update_schema_merge<<<h_bench.global_stack_index[1],1024>>>(d_bench, h_bench.global_stack_index[1]);
			check_execution();
			cudaDeviceSynchronize();
		}
		CUDA_SAFE_CALL(cudaMemcpy(&h_bench, d_bench, sizeof(workbench), cudaMemcpyDeviceToHost));
		logt("schema update %d grids", start, h_bench.grids_stack_index);
	}

	// clean the device bench for next round of checking
	cuda_cleargrids<<<bench->grids_stack_capacity/1024+1,1024>>>(d_bench);
	cuda_reset_bench<<<1,1>>>(d_bench);
}
