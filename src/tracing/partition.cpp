/*
 * partitioner.cpp
 *
 *  Created on: Feb 1, 2021
 *      Author: teng
 */

#include "trace.h"

workbench *partitioner::build_schema(Point *points, size_t num_objects){
	struct timeval start = get_cur_time();
	config->x_buffer = config->reach_distance*degree_per_meter_longitude(mbr.low[1]);
	config->y_buffer = config->reach_distance*degree_per_meter_latitude;

	// construct the QTree
	// todo parallelize it
	QTNode *qtree = new QTNode(mbr, config, points);
	for(uint pid=0;pid<num_objects;pid++){
		//log("%d",pid);
		qtree->insert(pid);
	}

	// set the ids and other stuff
	qtree->finalize();
	//qtree->print();


	// create and initialize the workbench
	workbench *bench = new workbench(config);

	bench->schema_counter = qtree->node_count();
	bench->grids_counter = qtree->leaf_count();
	assert(bench->schema_counter<bench->schema_capacity);
	assert(bench->grids_counter<bench->grids_capacity);

	bench->claim_space();

	// construct the schema with the QTree
	uint offset = 0;
	qtree->create_schema(bench->schema, offset);
	delete qtree;
	logt("partitioning schema is with %d grids",start,bench->grids_counter);
	return bench;
}

bool workbench::insert(uint curnode, uint pid){
	assert(schema[curnode].isleaf);
	uint gid = schema[curnode].node_id;
	assert(gid<grids_counter);
	lock(gid);
	uint cur_size = grid_counter[gid]++;
	// todo handle overflow
	if(cur_size<config->grid_capacity){
		grids[config->grid_capacity*gid+cur_size] = pid;
	}
	unlock(gid);
	// first batch of lookup pairs, start from offset 0
	grid_check[pid].pid = pid;
	grid_check[pid].gid = gid;
	grid_check[pid].offset = 0;
	grid_check[pid].inside = true;

	// is this point too close to the border?
	Point *p = points+pid;
	if(p->x+config->x_buffer>schema[curnode].mbr.high[0]||
	   p->y+config->y_buffer>schema[curnode].mbr.high[1]){
		lock();
		lookup_stack[0][stack_index[0]*2] = pid;
		lookup_stack[0][stack_index[0]*2+1] = 0;
		stack_index[0]++;
		unlock();
	}
	return true;
}

bool workbench::batch_insert(uint gid, uint num_objects, uint *pids){
	assert(num_objects<config->grid_capacity);
	// can only batch insert to an empty grid
	assert(grid_counter[gid]==0);
	lock(gid);
	memcpy(grids+config->grid_capacity*gid,pids,num_objects*sizeof(uint));
	grid_counter[gid] += num_objects;
	unlock(gid);
	return true;
}

// single thread function for assigning objects into grids following certain schema
void *partition_unit(void *arg){
	query_context *qctx = (query_context *)arg;
	workbench *bench = (workbench *)qctx->target[0];

	// pick one batch of point-grid pair for processing
	size_t start = 0;
	size_t end = 0;
	while(qctx->next_batch(start,end)){
		for(uint pid=start;pid<end;pid++){
			uint curnode = 0;
			Point *p = bench->points+pid;
			while(true){
				// assign to a child 0-3
				int child = (p->y>bench->schema[curnode].mid_y)*2+(p->x>bench->schema[curnode].mid_x);
				curnode = bench->schema[curnode].children[child];
				// is leaf
				if(bench->schema[curnode].isleaf){
					break;
				}
			}
			// pid belongs to such node
			bench->insert(curnode, pid);
		}
	}
	return NULL;
}



void workbench::partition(){
	// the schema has to be built
	struct timeval start = get_cur_time();

	// partitioning current batch of objects with the existing schema
	pthread_t threads[config->num_threads];
	query_context qctx;
	qctx.config = config;
	qctx.num_units = config->num_objects;
	qctx.target[0] = (void *)this;

	for(int i=0;i<config->num_threads;i++){
		pthread_create(&threads[i], NULL, partition_unit, (void *)&qctx);
	}
	for(int i = 0; i < config->num_threads; i++ ){
		void *status;
		pthread_join(threads[i], &status);
	}
	grid_check_counter = config->num_objects;
	logt("partition data: %d boundary points", start,stack_index[0]);
}
