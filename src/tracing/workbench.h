/*
 * working_bench.h
 *
 *  Created on: Feb 18, 2021
 *      Author: teng
 */

#ifndef SRC_TRACING_WORKBENCH_H_
#define SRC_TRACING_WORKBENCH_H_
#include "../util/util.h"
#include "../util/config.h"
#include "../util/query_context.h"
#include "../geometry/geometry.h"
#include "../index/QTree.h"

typedef struct checking_unit{
	uint pid;
	uint gid;
	unsigned short offset;
	unsigned short inside;
}checking_unit;

typedef struct meeting_unit{
	uint pid1;
	uint pid2;
	unsigned short start;
}meeting_unit;

typedef struct reach_unit{
	uint pid1;
	uint pid2;
}reach_unit;

// the workbench where stores the memory space
// used for processing


class workbench{
	pthread_mutex_t *insert_lk;
	void *data[100];
	size_t data_size[100];
	uint data_index = 0;
	void *allocate(size_t size);
public:
	uint test_counter = 0;
	configuration *config = NULL;
	uint cur_time = 0;

	// the pool of maintaining objects assignment
	// each grid buffer: |point_id1...point_idn|
	uint *grids = NULL;
	uint grid_capacity = 0;
	uint *grid_counter = NULL;

	// the stack that keeps the available grids
	uint *grids_stack = NULL;
	uint grids_stack_capacity = 0;
	uint grids_stack_index = 0;

	// the QTree schema
	QTSchema *schema = NULL;
	// stack that keeps the available schema nodes
	uint *schema_stack = NULL;
	uint schema_stack_capacity = 0;
	uint schema_stack_index = 0;

	// the space for point-unit pairs
	checking_unit *grid_check = NULL;
	uint grid_check_capacity = 0;
	uint grid_check_counter = 0;

	// the space for the overall meeting information maintaining now
	uint current_bucket = 0;
	meeting_unit *meeting_buckets[2] = {NULL,NULL};
	uint *meeting_buckets_counter[2] = {NULL,NULL};
	uint meeting_bucket_capacity = 0;

	// the space for the valid meeting information now
	meeting_unit *meetings = NULL;
	uint meeting_capacity = 0;
	uint meeting_counter = 0;

	// the processing stack for looking up
	uint *global_stack[2] = {NULL, NULL};
	uint global_stack_index[2] = {0,0};
	uint global_stack_capacity = 0;

	// external source
	Point *points = NULL;

	workbench(workbench *bench);
	workbench(configuration *conf);
	~workbench(){};
	void clear();

	// insert point pid to grid gid
	bool insert(uint gid, uint pid);
	bool batch_insert(uint gid, uint num_objects, uint *pids);

	// generate pid-gid-offset pairs for processing
	bool check(uint gid, uint pid);
	bool batch_check(checking_unit *cu, uint num);

	void merge_node(uint cur_node);
	void split_node(uint cur_node);
	void partition();
	void update_schema();
	void reachability();

	void claim_space();
	void reset(){
		// reset the number of objects in each grid
		for(int i=0;i<grids_stack_capacity;i++){
			grid_counter[i] = 0;
		}
		grid_check_counter = 0;
		global_stack_index[0] = 0;
		global_stack_index[1] = 0;
	}
	inline uint get_grid_size(uint gid){
		assert(gid<grids_stack_capacity);
		return min(grid_counter[gid],grid_capacity);
	}
	inline uint *get_grid(uint gid){
		assert(gid<grids_stack_capacity);
		return grids + gid*grid_capacity;
	}

	void sort_meeting();
	void update_meetings();
	void analyze_grids();
	void analyze_reaches();
	void analyze_meetings();
	void analyze_meeting_buckets();


	void lock(uint key = 0){
		pthread_mutex_lock(&insert_lk[key%MAX_LOCK_NUM]);
	}
	void unlock(uint key = 0){
		pthread_mutex_unlock(&insert_lk[key%MAX_LOCK_NUM]);
	}
};
extern void lookup_rec(QTSchema *schema, Point *p, uint curnode, vector<uint> &gids, double max_dist, bool include_owner = false);

#endif /* SRC_TRACING_WORKBENCH_H_ */
