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
	uint offset;
	bool inside;
}checking_unit;

typedef struct meeting_unit{
	uint pid1;
	uint pid2;
	uint start;
	uint end;
}meeting_unit;

typedef struct reach_unit{
	uint pid1;
	uint pid2;
}reach_unit;

// the workbench where stores the memory space
// used for processing

#define MAX_LOCK_NUM 100

class workbench{
	pthread_mutex_t insert_lk[MAX_LOCK_NUM];
public:
	configuration *config = NULL;
	uint cur_time = 0;

	// the pool of maintaining objects assignment
	// each grid buffer: |num_objects|point_id1...point_idn|
	uint *grids = NULL;
	uint num_grids = 0;

	// the space for point-unit pairs
	checking_unit *unit_lookup = NULL;
	uint unit_lookup_capacity = 0;
	uint unit_lookup_counter = 0;

	// the space for a reach unit in one time point
	reach_unit *reaches = NULL;
	uint reaches_capacity = 0;
	uint reaches_counter = 0;

	// the space for the overall meeting information maintaining now
	meeting_unit *meeting_buckets = NULL;
	uint meeting_bucket_capacity = 0;
	uint *meeting_buckets_counter = NULL;
	uint *meeting_buckets_counter_tmp = NULL;

	// the space for the valid meeting information now
	meeting_unit *meetings = NULL;
	uint meeting_capacity = 0;
	uint meeting_counter = 0;

	// the QTree schema
	QTSchema *schema = NULL;
	uint num_nodes = 0;

	// the processing stack for looking up
	uint *lookup_stack[2] = {NULL, NULL};
	uint stack_index[2] = {0,0};
	uint stack_capacity = 0;

	// external source
	Point *points = NULL;

	workbench(workbench *bench);
	workbench(configuration *conf);
	~workbench(){};
	void clear();

	// insert point pid to grid gid
	bool insert(uint gid, uint pid);
	//bool batch_insert(uint gid, uint num_objects, uint *pids);

	// generate pid-gid-offset pairs for processing
	bool check(uint gid, uint pid);
	bool batch_check(checking_unit *cu, uint num);

	void partition();
	void lookup();
	void reachability();

	bool batch_reach(reach_unit *ru, uint num);
	bool batch_meet(meeting_unit *mu, uint num);

	void claim_space(uint ng);
	void reset(){
		// reset the number of objects in each grid
		for(int i=0;i<num_grids;i++){
			grids[i*(config->grid_capacity+1)] = 0;
		}
		unit_lookup_counter = 0;
		reaches_counter = 0;
		stack_index[0] = 0;
		stack_index[1] = 0;
	}
	inline uint get_grid_size(uint gid){
		assert(gid<num_grids);
		return min(grids[gid*(config->grid_capacity+1)],config->grid_capacity);
	}
	inline uint *get_grid(uint gid){
		assert(gid<num_grids);
		return grids + gid*(config->grid_capacity+1)+1;
	}

	void update_meetings();
	void compact_meetings();


	void analyze_meetings();
	void analyze_grids();
	void analyze_checkings();

	void lock(uint key = 0){
		pthread_mutex_lock(&insert_lk[key%MAX_LOCK_NUM]);
	}
	void unlock(uint key = 0){
		pthread_mutex_unlock(&insert_lk[key%MAX_LOCK_NUM]);
	}
};


#endif /* SRC_TRACING_WORKBENCH_H_ */
