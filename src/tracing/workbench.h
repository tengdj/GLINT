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
}checking_unit;

typedef struct meeting_unit{
	uint pid1;
	uint pid2;
	uint start;
	uint end;
}meeting_unit;

// the workbench where stores the memory space
// used for processing

class workbench{
	pthread_mutex_t insert_lk[50];
public:
	configuration config;

	// the pool of maintaining objects assignment
	// each grid buffer: |num_objects|point_id1...point_idn|
	uint *grids = NULL;
	size_t num_grids = 0;

	meeting_unit *meetings = NULL;
	size_t meeting_capacity = 0;
	size_t num_meeting = 0;

	// the QTree schema
	QTSchema *schema = NULL;
	uint num_nodes = 0;

	// the space for storing the point-grid pairs
	checking_unit *checking_units = NULL;
	uint num_checking_units = 0;
	size_t checking_units_capacity = 0;

	// the processing stack for looking up
	uint *lookup_stack[2] = {NULL, NULL};
	uint stack_index[2] = {0,0};

	// external source
	Point *points = NULL;

	workbench(configuration conf);
	~workbench();

	// insert point pid to grid gid
	bool insert(uint gid, uint pid);
	bool batch_insert(uint gid, uint num_objects, uint *pids);

	// generate pid-gid-offset pairs for processing
	bool check(uint gid, uint pid);
	bool batch_check(checking_unit *cu, uint num_cu);


	void claim_space(uint ng);
	void reset(){
		// reset the number of objects in each grid
		for(int i=0;i<num_grids;i++){
			grids[i*(config.grid_capacity+1)] = 0;
		}
		num_checking_units = 0;
		num_meeting = 0;
	}
	inline uint get_grid_size(uint gid){
		return grids[gid*(config.grid_capacity+1)];
	}
	inline uint *get_grid(uint gid){
		return grids + gid*(config.grid_capacity+1)+1;
	}
	inline void reset_grid(uint gid){
		grids[gid*(config.grid_capacity+1)] = 0;
	}
	inline void increase_grid_size(uint gid){
		grids[gid*(config.grid_capacity+1)]++;
	}
	void analyze_meetings();
	void analyze_grids();
	void analyze_checkings();


};


#endif /* SRC_TRACING_WORKBENCH_H_ */
