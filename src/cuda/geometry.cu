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




/*
 *
 * kernel functions
 *
 * */

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
	bench->meeting_counter = 0;
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


#define PER_STACK_SIZE 5

__global__
void cuda_pack_lookup(workbench *bench){
	int pid = blockIdx.x*blockDim.x+threadIdx.x;
	if(pid>=bench->config->num_objects){
		return;
	}

	uint idx = atomicAdd(&bench->global_stack_index[0],1);
	assert(idx<bench->global_stack_capacity);
	bench->global_stack[0][idx*2] = pid;
	bench->global_stack[0][idx*2+1] = 0;
}

__global__
void cuda_reset_stack(workbench *bench, uint batch_size){
	int cur_idx = blockIdx.x*blockDim.x+threadIdx.x;
	if(cur_idx>=batch_size){
		return;
	}
	int block_stack_size = 2*1024*PER_STACK_SIZE;
	int *cur_stack_idx = (int *)bench->global_stack[1]+blockIdx.x*block_stack_size;
	*cur_stack_idx = 0;
}

__global__
void cuda_lookup_block(workbench *bench, int start_idx, int batch_size, bool include_contain){

	int cur_idx = blockIdx.x*blockDim.x+threadIdx.x;
	int idx = cur_idx + start_idx;
	if(cur_idx>=batch_size){
		return;
	}
	int pid = bench->global_stack[0][idx*2];
	int nodeid = bench->global_stack[0][idx*2+1];

	int block_stack_size = 2*1024*PER_STACK_SIZE;

	//printf("%d %d %d %d %d\n",threadIdx.x,warp_id,batch_size,warp_id*warp_stack_size, bench->global_stack_capacity);
	assert(blockIdx.x*block_stack_size<bench->global_stack_capacity);

	int *cur_stack_idx = (int *)bench->global_stack[1]+blockIdx.x*block_stack_size;
	int *cur_worker_idx = (int *)bench->global_stack[1]+blockIdx.x*block_stack_size+1;

	uint *cur_stack = bench->global_stack[1]+blockIdx.x*block_stack_size+2;
	*cur_stack_idx = 0;
	*cur_worker_idx = 0;
	__syncthreads();

	int stack_index = atomicAdd(cur_stack_idx, 1);
	cur_stack[2*stack_index] = pid;
	cur_stack[2*stack_index+1] = nodeid;

	//printf("%d:\tinit push %d\n",threadIdx.x,stack_index);
	__syncthreads();

	while(true){
		bool busy = false;
		stack_index = atomicSub(cur_stack_idx, 1)-1;
		//printf("%d:\tpop %d\n",threadIdx.x, stack_index);
		__syncthreads();
		if(stack_index<0){
			stack_index = atomicAdd(cur_stack_idx, 1);
			//printf("%d:\tinc %d\n",threadIdx.x, stack_index);
		}else{
			busy = true;
			atomicAdd(cur_worker_idx, 1);
		}
		__syncthreads();

		//printf("num workers: %d\n",*cur_worker_idx);
		if(*cur_worker_idx==0){
			break;
		}
		if(busy){

			uint pid = cur_stack[2*stack_index];
			uint curnode = cur_stack[2*stack_index+1];
			Point *p = bench->points+pid;
			//printf("process: %d %d %d\n",stack_index,pid,curnode);

			for(int i=0;i<4;i++){
				uint child_offset = bench->schema[curnode].children[i];
				double dist = distance(&bench->schema[child_offset].mbr, p);
				if(dist<=bench->config->reach_distance){
					if(bench->schema[child_offset].type==LEAF){
						uint gid = bench->schema[child_offset].grid_id;
						assert(gid<bench->grids_stack_capacity);
						if(include_contain&&contain(&bench->schema[child_offset].mbr,p)){
							uint *cur_grid = bench->grids+bench->grid_capacity*gid;
							uint cur_loc = atomicAdd(bench->grid_counter+gid,1);
							if(cur_loc<bench->grid_capacity){
								*(cur_grid+cur_loc) = pid;
							}
							uint glid = atomicAdd(&bench->grid_check_counter,1);
							assert(glid<bench->grid_check_capacity);
							bench->grid_check[glid].pid = pid;
							bench->grid_check[glid].gid = gid;
							bench->grid_check[glid].offset = 0;
							bench->grid_check[glid].inside = true;
						}else if(p->y<bench->schema[child_offset].mbr.low[1]||
						   (p->y<bench->schema[child_offset].mbr.high[1]
							&& p->x<bench->schema[child_offset].mbr.low[0])){
							uint glid = atomicAdd(&bench->grid_check_counter,1);
							assert(glid<bench->grid_check_capacity);
							bench->grid_check[glid].pid = pid;
							bench->grid_check[glid].gid = gid;
							bench->grid_check[glid].offset = 0;
							bench->grid_check[glid].inside = false;
						}
					}else{
						stack_index = atomicAdd(cur_stack_idx, 1);
						//printf("%d:\tnew push %d\n",threadIdx.x,stack_index);
						assert(stack_index<PER_STACK_SIZE*1024);
						cur_stack[2*stack_index] = pid;
						cur_stack[2*stack_index+1] = child_offset;
					}
				}
			}
			atomicSub(cur_worker_idx, 1);
		}
		__syncthreads();
	}
	if(idx == 0){
		bench->global_stack_index[0] = 0;
	}
}



__global__
void cuda_unroll(workbench *bench, uint inistial_size){
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
inline size_t mhash64( size_t k ){
  k ^= k >> 33;
  k *= 0xff51afd7ed558ccdULL;
  k ^= k >> 33;
  k *= 0xc4ceb9fe1a85ec53ULL;
  k ^= k >> 33;
  return k;
}

__device__
uint mhash32(uint k)
{
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;
    return k;
}

#define ULL_MAX (size_t)1<<62


__global__
void cuda_clean_buckets(workbench *bench){
	int bid = blockIdx.x*blockDim.x+threadIdx.x;
	if(bid>=bench->config->num_meeting_buckets){
		return;
	}
	meeting_unit *new_bucket = bench->meeting_buckets[0]+bid*bench->config->bucket_size;
	meeting_unit *old_bucket = bench->meeting_buckets[1]+bid*bench->config->bucket_size;

	for(int i=0;i<bench->config->bucket_size;i++){
		new_bucket[i].key = ULL_MAX;
		old_bucket[i].key = ULL_MAX;
	}
}

__global__
void cuda_refinement(workbench *bench){

	// the objects in which grid need be processed
	int loc = threadIdx.y;
	int pairid = blockIdx.x*blockDim.x+threadIdx.x;
	if(pairid>=bench->grid_check_counter){
		return;
	}

	uint gid = bench->grid_check[pairid].gid;
	uint offset = bench->grid_check[pairid].offset;

	uint size = min(bench->grid_counter[gid],bench->grid_capacity)-offset;
	if(bench->config->unroll && size>bench->config->zone_capacity){
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
			uint pid1 = min(pid,target_pid);
			uint pid2 = max(target_pid,pid);
			//size_t key = (((size_t)pid1)<<32)+pid2;
			size_t key = (pid1+pid2)*(pid1+pid2+1)/2+pid2;

			meeting_unit *mtable = bench->meeting_buckets[bench->current_bucket];

			if(bench->config->use_hash){
				size_t kHashTableCapacity = ((size_t)bench->config->bucket_size*bench->config->num_meeting_buckets);
				size_t slot = key%kHashTableCapacity;
				int ited = 0;
				//printf("%ld %ld %ld\n",slot,key,kHashTableCapacity);
				while (true){
					unsigned long long prev = atomicCAS((unsigned long long *)&mtable[slot].key, ULL_MAX, (unsigned long long)key);
					//printf("%ld\n",prev,ULL_MAX,bench->meeting_buckets[bench->current_bucket][slot].key);
					if (prev == ULL_MAX){
						mtable[slot].key = key;
						mtable[slot].pid1 = pid1;
						mtable[slot].pid2 = pid2;
						mtable[slot].start = bench->cur_time;
						break;
					}
					slot = (slot + 1)%kHashTableCapacity;
					ited++;
//					if(ited>5){
//						printf("%ld %ld %d\n",slot,key,ited);
//						break;
//					}
				}
				if(ited>0){
					//atomicAdd(&bench->meeting_counter,1);
				}
			}else{
				//uint bid = key%bench->config->num_meeting_buckets;
				//printf("%d\n",bid);
				uint bid = (pid+target_pid)%bench->config->num_meeting_buckets;
				uint loc = atomicAdd(bench->meeting_buckets_counter[bench->current_bucket]+bid,1);
				// todo handling overflow
				if(loc<bench->config->bucket_size){
					meeting_unit *bucket = mtable+bid*bench->config->bucket_size;
					bucket[loc].pid1 = pid1;
					bucket[loc].pid2 = pid2;
					bucket[loc].start = bench->cur_time;
				}
			}
		}
	}
}

__global__
void cuda_identify_meetings_hash(workbench *bench){

	int bid = blockIdx.x*blockDim.x+threadIdx.x;
	size_t kHashTableCapacity = bench->config->num_meeting_buckets*bench->config->bucket_size;
	if(bid>=kHashTableCapacity){
		return;
	}
	meeting_unit *meet = bench->meeting_buckets[!bench->current_bucket]+bid;
	meeting_unit *new_tab = bench->meeting_buckets[bench->current_bucket];
	if(meet->key==ULL_MAX){
		return;
	}
	size_t slot = meet->key%kHashTableCapacity;

	int ite = 0;
	while (ite++<6){
		if (new_tab[slot].key == meet->key){
			new_tab[slot].start = meet->start;
			atomicAdd(&bench->meeting_counter,1);
			return;
		}
		if (new_tab[slot].key == ULL_MAX){
			break;
		}
		slot = (slot + 1)%kHashTableCapacity;
	}
	if(bench->cur_time - meet->start>=bench->config->min_meet_time){
//		uint meeting_idx = atomicAdd(&bench->meeting_counter,1);
//		if(meeting_idx<bench->meeting_capacity){
//			bench->meetings[meeting_idx] = *meet;
//		}
	}
}



/* This function takes last element as pivot, places
   the pivot element at its correct position in sorted
    array, and places all smaller (smaller than pivot)
   to left of pivot and all greater elements to right
   of pivot */
__device__
inline int partition(meeting_unit *arr, int low, int high){
    int pivot = arr[high].pid1;    // pivot
    int i = (low - 1);  // Index of smaller element

    meeting_unit tmp;
    for (int j = low; j <= high- 1; j++){
        // If current element is smaller than or
        // equal to pivot
        if (arr[j].pid1 <= pivot){
            i++;    // increment index of smaller element
            tmp = arr[i];
            arr[i] = arr[j];
            arr[j] = tmp;
        }
    }
    tmp = arr[i + 1];
    arr[i + 1] = arr[high];
    arr[high] = tmp;
    return (i + 1);
}


/* The main function that implements QuickSort
 arr[] --> Array to be sorted,
  low  --> Starting index,
  high  --> Ending index */
__device__
inline void quickSort(meeting_unit *arr, int low, int high){
    if (low < high){
        /* pi is partitioning index, arr[p] is now
           at right place */
        int pi = partition(arr, low, high);
        // Separately sort elements before
        // partition and after partition
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

__device__
inline bool isvalid(int p,int r, int size){
	return 0 <= p && p< r &&r < size;
}
__global__
void cuda_sort_meetings(workbench *bench){

	int bid = blockIdx.x*blockDim.x+threadIdx.x;
	if(bid>=bench->config->num_meeting_buckets){
		return;
	}
	meeting_unit *bucket = bench->meeting_buckets[bench->current_bucket]+bid*bench->config->bucket_size;
	int size = min(bench->meeting_buckets_counter[bench->current_bucket][bid],bench->config->bucket_size);
	int stack_size = 4*bench->global_stack_capacity/bench->config->num_meeting_buckets;
	int *cur_stack = (int *)(bench->global_stack[0] + stack_size*bid);
	int stack_index = 0;
	if(size>1){
		cur_stack[stack_index++] = 0;
		cur_stack[stack_index++] = size - 1;
		while(stack_index){
			uint q = cur_stack[--stack_index];
			uint p = cur_stack[--stack_index];
			uint r = partition(bucket,p,q);
			if(isvalid(p,r-1,size)){
				cur_stack[stack_index++] = p;
				cur_stack[stack_index++] = r-1;
			}
			if(isvalid(r+1,q,size)){
				cur_stack[stack_index++] = r+1;
				cur_stack[stack_index++] = q;
			}
			assert(stack_index<stack_size);
		}
	}
//	if(bid==0){
//		for(int i=0;i<size;i++){
//			printf("%d ",bucket[i].pid1);
//		}
//		printf("\n");
//	}
}

/*
 * update the first meet
 * */
__global__
void cuda_identify_meetings_sort(workbench *bench){

	int bid = blockIdx.x*blockDim.x+threadIdx.x;
	if(bid>=bench->config->num_meeting_buckets){
		return;
	}

	meeting_unit *bucket_new = bench->meeting_buckets[bench->current_bucket]+bid*bench->config->bucket_size;
	meeting_unit *bucket_old = bench->meeting_buckets[!bench->current_bucket]+bid*bench->config->bucket_size;

	uint size_new = bench->meeting_buckets_counter[bench->current_bucket][bid];
	uint size_old = bench->meeting_buckets_counter[!bench->current_bucket][bid];

	uint i=0,j=0;
	for(;i<size_old&&i<bench->config->bucket_size;i++){
		bool updated = false;
		for(;j<size_new&&j<bench->config->bucket_size;j++){
			if(bucket_old[j].pid1==bucket_new[i].pid1){
				bucket_new[i].start = bucket_old[i].start;
				updated = true;
				break;
			}else if(bucket_old[i].pid1>bucket_new[j].pid1){
				break;
			}
		}
		// the old meeting is over
		if(!updated&&
			bench->cur_time - bucket_old[i].start>=bench->config->min_meet_time){
			uint meeting_idx = atomicAdd(&bench->meeting_counter,1);
			if(meeting_idx<bench->meeting_capacity){
				bench->meetings[meeting_idx] = bucket_old[i];
			}
		}
	}
	// reset the old buckets for next batch of processing
	bench->meeting_buckets_counter[!bench->current_bucket][bid] = 0;
}

/*
 * update the first meet
 * */
__global__
void cuda_identify_meetings(workbench *bench){

	int bid = blockIdx.x*blockDim.x+threadIdx.x;
	if(bid>=bench->config->num_meeting_buckets){
		return;
	}

	meeting_unit *bucket_new = bench->meeting_buckets[bench->current_bucket]+bid*bench->config->bucket_size;
	meeting_unit *bucket_old = bench->meeting_buckets[!bench->current_bucket]+bid*bench->config->bucket_size;

	uint size_new = bench->meeting_buckets_counter[bench->current_bucket][bid];
	uint size_old = bench->meeting_buckets_counter[!bench->current_bucket][bid];

	for(uint i=0;i<size_old&&i<bench->config->bucket_size;i++){
		bool updated = false;
		for(uint j=0;j<size_new&&j<bench->config->bucket_size;j++){
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
			if(meeting_idx<bench->meeting_capacity){
				bench->meetings[meeting_idx] = bucket_old[i];
			}
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
	log("GPU memory:");
	struct timeval start = get_cur_time();
	gpu->clear();
	// use h_bench as a container to copy in and out GPU
	workbench h_bench(bench);
	// space for the raw points data
	h_bench.points = (Point *)gpu->allocate(bench->config->num_objects*sizeof(Point));
	size_t size = bench->config->num_objects*sizeof(Point);
	log("\t%.2f MB\tpoints",1.0*size/1024/1024);

	// space for the pids of all the grids
	h_bench.grids = (uint *)gpu->allocate(bench->grids_stack_capacity*bench->grid_capacity*sizeof(uint));
	h_bench.grid_counter = (uint *)gpu->allocate(bench->grids_stack_capacity*sizeof(uint));
	h_bench.grids_stack = (uint *)gpu->allocate(bench->grids_stack_capacity*sizeof(uint));
	size = bench->grids_stack_capacity*bench->grid_capacity*sizeof(uint)+bench->grids_stack_capacity*sizeof(uint)+bench->grids_stack_capacity*sizeof(uint);
	log("\t%.2f MB\tgrids",1.0*size/1024/1024);

	// space for the QTtree schema
	h_bench.schema = (QTSchema *)gpu->allocate(bench->schema_stack_capacity*sizeof(QTSchema));
	h_bench.schema_stack = (uint *)gpu->allocate(bench->schema_stack_capacity*sizeof(uint));
	size = bench->schema_stack_capacity*sizeof(QTSchema)+bench->schema_stack_capacity*sizeof(uint);
	log("\t%.2f MB\tschema",1.0*size/1024/1024);

	// space for the pid-zid pairs
	h_bench.grid_check = (checking_unit *)gpu->allocate(bench->grid_check_capacity*sizeof(checking_unit));
	size = bench->grid_check_capacity*sizeof(checking_unit);
	log("\t%.2f MB\trefine list",1.0*size/1024/1024);

	// space for processing stack
	h_bench.global_stack[0] = (uint *)gpu->allocate(bench->global_stack_capacity*4*sizeof(uint));
	h_bench.global_stack[1] = h_bench.global_stack[0]+bench->global_stack_capacity*2;
	size = 2*bench->global_stack_capacity*2*sizeof(uint);
	log("\t%.2f MB\tstack",1.0*size/1024/1024);

	h_bench.meeting_buckets[0] = (meeting_unit *)gpu->allocate(bench->config->num_meeting_buckets*bench->config->bucket_size*sizeof(meeting_unit));
	h_bench.meeting_buckets[1] = (meeting_unit *)gpu->allocate(bench->config->num_meeting_buckets*bench->config->bucket_size*sizeof(meeting_unit));
	h_bench.meeting_buckets_counter[0] = (uint *)gpu->allocate(bench->config->num_meeting_buckets*sizeof(uint));
	h_bench.meeting_buckets_counter[1] = (uint *)gpu->allocate(bench->config->num_meeting_buckets*sizeof(uint));
	size = 2*(bench->config->num_meeting_buckets*bench->config->bucket_size*sizeof(meeting_unit)+bench->config->num_meeting_buckets*sizeof(uint));
	log("\t%.2f MB\thash table",1.0*size/1024/1024);

	h_bench.meetings = (meeting_unit *)gpu->allocate(bench->meeting_capacity*sizeof(meeting_unit));
	size = bench->meeting_capacity*sizeof(meeting_unit);
	log("\t%.2f MB\tmeetings",1.0*size/1024/1024);


	// space for the configuration
	h_bench.config = (configuration *)gpu->allocate(sizeof(configuration));
	// space for the mapping of bench in GPU
	workbench *d_bench = (workbench *)gpu->allocate(sizeof(workbench));

	// the configuration and schema are fixed
	CUDA_SAFE_CALL(cudaMemcpy(h_bench.schema, bench->schema, bench->schema_stack_capacity*sizeof(QTSchema), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(h_bench.config, bench->config, sizeof(configuration), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_bench, &h_bench, sizeof(workbench), cudaMemcpyHostToDevice));

	cuda_init_grids_stack<<<bench->grids_stack_capacity/1024, 1024>>>(d_bench);
	cuda_init_schema_stack<<<bench->schema_stack_capacity/1024, 1024>>>(d_bench);
	cuda_clean_buckets<<<bench->config->num_meeting_buckets/1024+1,1024>>>(d_bench);

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
	bench->pro.copy_time += get_time_elapsed(start,false);
	logt("copy in data", start);

	if(bench->config->phased_lookup){
		// do the partition
		cuda_partition<<<bench->config->num_objects/1024+1,1024>>>(d_bench);
		check_execution();
		cudaDeviceSynchronize();
		CUDA_SAFE_CALL(cudaMemcpy(&h_bench, d_bench, sizeof(workbench), cudaMemcpyDeviceToHost));
		bench->pro.filter_time += get_time_elapsed(start,false);
		logt("partition data %d still need lookup", start,h_bench.global_stack_index[0]);
	}else{
		cuda_pack_lookup<<<bench->config->num_objects/1024+1,1024>>>(d_bench);
		check_execution();
		cudaDeviceSynchronize();
		CUDA_SAFE_CALL(cudaMemcpy(&h_bench, d_bench, sizeof(workbench), cudaMemcpyDeviceToHost));
	}

	// do the tree lookup
	uint batch_size = bench->global_stack_capacity/PER_STACK_SIZE/2;
	for(int i=0;i<h_bench.global_stack_index[0];i+=batch_size){
		int bs = min(batch_size,h_bench.global_stack_index[0]-i);
		cuda_reset_stack<<<bs/1024+1,1024>>>(d_bench, bs);
		cuda_lookup_block<<<bs/1024+1,1024>>>(d_bench, i, bs, !bench->config->phased_lookup);
		check_execution();
		cudaDeviceSynchronize();
	}
	CUDA_SAFE_CALL(cudaMemcpy(&h_bench, d_bench, sizeof(workbench), cudaMemcpyDeviceToHost));
	bench->pro.filter_time += get_time_elapsed(start,false);
	logt("filtering with %d checkings", start,h_bench.grid_check_counter);

	// do the unrollment
	if(bench->config->unroll){
		cuda_unroll<<<h_bench.grid_check_counter/1024+1,1024>>>(d_bench,h_bench.grid_check_counter);
		check_execution();
		cudaDeviceSynchronize();
		CUDA_SAFE_CALL(cudaMemcpy(&h_bench, d_bench, sizeof(workbench), cudaMemcpyDeviceToHost));
		bench->pro.refine_time += get_time_elapsed(start,false);
		logt("%d pid-grid-offset tuples need be checked", start,h_bench.grid_check_counter);
	}
	bench->grid_check_counter = h_bench.grid_check_counter;


	// compute the reachability of objects in each partitions

	uint thread_y = bench->config->unroll?bench->config->zone_capacity:bench->grid_capacity;
	uint thread_x = 1024/thread_y;
	dim3 block(thread_x, thread_y);
	cuda_refinement<<<h_bench.grid_check_counter/thread_x+1,block>>>(d_bench);
	check_execution();
	cudaDeviceSynchronize();
	CUDA_SAFE_CALL(cudaMemcpy(&h_bench, d_bench, sizeof(workbench), cudaMemcpyDeviceToHost));
	bench->pro.refine_time += get_time_elapsed(start,false);
	logt("reaches computation %d", start,h_bench.meeting_counter);

	// update the meeting hash table
	uint origin_num_meeting = h_bench.meeting_counter;

	if(bench->config->use_hash){
		size_t kHashTableCapacity = bench->config->num_meeting_buckets*bench->config->bucket_size;
		cuda_identify_meetings_hash<<<kHashTableCapacity/1024+1,1024>>>(d_bench);
	}else if(bench->config->brute_meeting){
		cuda_identify_meetings<<<bench->config->num_meeting_buckets/1024+1,1024>>>(d_bench);
	}else{
		cuda_sort_meetings<<<bench->config->num_meeting_buckets/1024+1,1024>>>(d_bench);
		check_execution();
		cudaDeviceSynchronize();
		bench->pro.meeting_update_time += get_time_elapsed(start,false);
		logt("sort buckets", start);
		cuda_identify_meetings_sort<<<bench->config->num_meeting_buckets/1024+1,1024>>>(d_bench);
	}
	check_execution();
	cudaDeviceSynchronize();
	CUDA_SAFE_CALL(cudaMemcpy(&h_bench, d_bench, sizeof(workbench), cudaMemcpyDeviceToHost));
	bench->pro.meeting_update_time += get_time_elapsed(start,false);
	logt("meeting buckets update %d new meetings found", start, h_bench.meeting_counter-origin_num_meeting);



	// todo do the data analyzes, for test only, should not copy out so much stuff
	do{
		if(bench->config->analyze_grid||bench->config->analyze_reach||bench->config->profile){
			CUDA_SAFE_CALL(cudaMemcpy(bench->grid_counter, h_bench.grid_counter,
					bench->grids_stack_capacity*sizeof(uint), cudaMemcpyDeviceToHost));
			logt("copy out grid counting data", start);
		}
		if(bench->config->analyze_reach){
			CUDA_SAFE_CALL(cudaMemcpy(bench->grids, h_bench.grids,
								bench->grids_stack_capacity*bench->grid_capacity*sizeof(uint), cudaMemcpyDeviceToHost));
			CUDA_SAFE_CALL(cudaMemcpy(bench->schema, h_bench.schema,
					bench->schema_stack_capacity*sizeof(QTSchema), cudaMemcpyDeviceToHost));
			CUDA_SAFE_CALL(cudaMemcpy(bench->meeting_buckets[bench->current_bucket], h_bench.meeting_buckets[bench->current_bucket],
								bench->config->num_meeting_buckets*bench->config->bucket_size*sizeof(meeting_unit), cudaMemcpyDeviceToHost));
			bench->schema_stack_index = h_bench.schema_stack_index;
			bench->grids_stack_index = h_bench.grids_stack_index;
			logt("copy out grid, schema, meeting buckets data", start);
		}
		if(bench->config->analyze_meeting||bench->config->analyze_reach||bench->config->profile){
			CUDA_SAFE_CALL(cudaMemcpy(bench->meeting_buckets_counter[bench->current_bucket], h_bench.meeting_buckets_counter[bench->current_bucket],
					bench->config->num_meeting_buckets*sizeof(uint), cudaMemcpyDeviceToHost));
			logt("copy out meeting buckets counting data", start);
		}
	}while(false);

	// do the schema update
	if(bench->config->dynamic_schema){
		// update the schema for future processing
		cuda_update_schema_collect<<<bench->schema_stack_capacity/1024+1,1024>>>(d_bench);
		check_execution();
		cudaDeviceSynchronize();
		CUDA_SAFE_CALL(cudaMemcpy(&h_bench, d_bench, sizeof(workbench), cudaMemcpyDeviceToHost));
		if(h_bench.global_stack_index[0]>0){
			cuda_update_schema_split<<<h_bench.global_stack_index[0]/1024+1,1024>>>(d_bench, h_bench.global_stack_index[0]);
			check_execution();
			cudaDeviceSynchronize();
		}
		if(h_bench.global_stack_index[1]>0){
			cuda_update_schema_merge<<<h_bench.global_stack_index[1]/1024+1,1024>>>(d_bench, h_bench.global_stack_index[1]);
			check_execution();
			cudaDeviceSynchronize();
		}
		CUDA_SAFE_CALL(cudaMemcpy(&h_bench, d_bench, sizeof(workbench), cudaMemcpyDeviceToHost));
		bench->pro.index_update_time += get_time_elapsed(start,false);
		logt("schema update %d grids", start, h_bench.grids_stack_index);
	}

	if(h_bench.meeting_counter>0){
		bench->meeting_counter = h_bench.meeting_counter;

		CUDA_SAFE_CALL(cudaMemcpy(bench->meetings, h_bench.meetings, min(bench->meeting_capacity, h_bench.meeting_counter)*sizeof(meeting_unit), cudaMemcpyDeviceToHost));
		bench->pro.copy_time += get_time_elapsed(start,false);
		logt("copy out %d meeting data", start,h_bench.meeting_counter);
	}
	// clean the device bench for next round of checking
	cuda_cleargrids<<<bench->grids_stack_capacity/1024+1,1024>>>(d_bench);
	cuda_reset_bench<<<1,1>>>(d_bench);
}
