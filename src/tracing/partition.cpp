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
	bench->claim_space();

	// initialize the schema stack
	for(int i=0;i<bench->schema_stack_capacity;i++){
		bench->schema_stack[i] = i;
	}
	// initialize the grid stack
	for(int i=0;i<bench->grids_stack_capacity;i++){
		bench->grids_stack[i] = i;
	}

	// pop the top grids and schema nodes
	bench->schema_stack_index += qtree->node_count();
	bench->grids_stack_index += qtree->leaf_count();
	assert(bench->schema_stack_index<bench->schema_stack_capacity);
	assert(bench->grids_stack_index<bench->grids_stack_capacity);
	// construct the schema with the QTree
	uint offset = 0;
	qtree->create_schema(bench->schema, offset);

	delete qtree;
	logt("partitioning schema is with %d grids",start,bench->grids_stack_index);
	return bench;
}

bool workbench::insert(uint curnode, uint pid){
	assert(schema[curnode].type==LEAF);
	uint gid = schema[curnode].grid_id;
	assert(gid<grids_stack_capacity);
	lock(gid);
	uint cur_size = grid_counter[gid]++;
	// todo handle overflow
	if(cur_size<grid_capacity){
		grids[grid_capacity*gid+cur_size] = pid;
	}
	unlock(gid);
	// first batch of lookup pairs, start from offset 0
	grid_check[pid].pid = pid;
	grid_check[pid].gid = gid;
	grid_check[pid].inside = true;

	// is this point too close to the border?
	Point *p = points+pid;
	if(p->x+config->x_buffer>schema[curnode].mbr.high[0]||
	   p->y+config->y_buffer>schema[curnode].mbr.high[1]){
		lock();
		lookup_stack[0][lookup_stack_index[0]*2] = pid;
		lookup_stack[0][lookup_stack_index[0]*2+1] = 0;
		lookup_stack_index[0]++;
		unlock();
	}
	return true;
}

bool workbench::batch_insert(uint gid, uint num_objects, uint *pids){
	assert(num_objects<grid_capacity);
	// can only batch insert to an empty grid
	assert(grid_counter[gid]==0);
	lock(gid);
	memcpy(grids+grid_capacity*gid,pids,num_objects*sizeof(uint));
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
				if(bench->schema[curnode].type==LEAF){
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
	// each point added one point-grid pair with offset 0
	grid_check_counter = config->num_objects;
	logt("partition data: %d boundary points", start,lookup_stack_index[0]);
}


void workbench::merge_node(uint cur_node){
	assert(schema[cur_node].type==BRANCH);
	lock();
	//printf("merge: %d\n",cur_node);

	//schema[cur_node].mbr.print();
	//reclaim the children
	for(int i=0;i<4;i++){
		uint child_offset = schema[cur_node].children[i];
		assert(schema[child_offset].type==LEAF);
		schema[child_offset].type = INVALID;
		//schema[child_offset].mbr.print();
		// push the schema and grid spaces to stack for reuse
		schema_stack[--schema_stack_index] = child_offset;
		grids_stack[--grids_stack_index] = schema[child_offset].grid_id;
		grid_counter[schema[child_offset].grid_id] = 0;
	}
	schema[cur_node].type = LEAF;
	schema[cur_node].grid_id = grids_stack[grids_stack_index++];
	unlock();
}

void workbench::split_node(uint cur_node){
	assert(schema[cur_node].type==LEAF);
	lock();

	//printf("split: %d\n",cur_node);
	//schema[cur_node].mbr.print();
	schema[cur_node].type = BRANCH;
	grids_stack[--grids_stack_index] = schema[cur_node].grid_id;

	double xhalf = schema[cur_node].mid_x-schema[cur_node].mbr.low[0];
	double yhalf = schema[cur_node].mid_y-schema[cur_node].mbr.low[1];

	for(int i=0;i<4;i++){
		// pop space for schema and grid
		assert(schema_stack_index<schema_stack_capacity);
		uint child = schema_stack[schema_stack_index++];
		schema[cur_node].children[i] = child;

		assert(grids_stack_index<grids_stack_capacity);
		schema[child].grid_id = grids_stack[grids_stack_index++];
		grid_counter[schema[child].grid_id] = 0;
		schema[child].level = schema[cur_node].level+1;
		assert(schema[child].type==INVALID);
		schema[child].type = LEAF;
		schema[child].overflow_count = 0;
		schema[child].underflow_count = 0;

		schema[child].mbr.low[0] = schema[cur_node].mbr.low[0]+(i%2==1)*xhalf;
		schema[child].mbr.low[1] = schema[cur_node].mbr.low[1]+(i/2==1)*yhalf;
		schema[child].mbr.high[0] = schema[cur_node].mid_x+(i%2==1)*xhalf;
		schema[child].mbr.high[1] = schema[cur_node].mid_y+(i/2==1)*yhalf;
		schema[child].mid_x = (schema[child].mbr.low[0]+schema[child].mbr.high[0])/2;
		schema[child].mid_y = (schema[child].mbr.low[1]+schema[child].mbr.high[1])/2;
		//schema[child].mbr.print();
	}

	unlock();
}

void *update_schema_unit(void *arg){

	query_context *qctx = (query_context *)arg;
	workbench *bench = (workbench *)qctx->target[0];

	// pick one batch of schema node for processing
	size_t start = 0;
	size_t end = 0;
	while(qctx->next_batch(start,end)){
		for(uint i=start;i<end;i++){
			uint curnode = bench->schema_stack[i];
			if(bench->schema[curnode].type==LEAF){
				if(bench->grid_counter[bench->schema[curnode].grid_id]>bench->config->grid_capacity){
					bench->schema[curnode].overflow_count++;
					// this node is overflowed a continuous number of times, split it
					if(bench->schema[curnode].overflow_count>=bench->config->schema_update_delay){
						bench->split_node(curnode);
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
					bench->schema[curnode].underflow_count++;
					// the children of this node need be deallocated
					if(bench->schema[curnode].underflow_count>=bench->config->schema_update_delay){
						bench->merge_node(curnode);
						bench->schema[curnode].underflow_count = 0;
					}
				}else{
					bench->schema[curnode].underflow_count = 0;
				}
			}
		}
	}
	return NULL;
}



void workbench::update_schema(){
	// the schema has to be built
	struct timeval start = get_cur_time();

	// partitioning current batch of objects with the existing schema
	pthread_t threads[config->num_threads];
	query_context qctx;
	qctx.config = config;
	qctx.num_units = schema_stack_capacity;
	qctx.target[0] = (void *)this;

	for(int i=0;i<config->num_threads;i++){
		pthread_create(&threads[i], NULL, update_schema_unit, (void *)&qctx);
	}
	for(int i = 0; i < config->num_threads; i++ ){
		void *status;
		pthread_join(threads[i], &status);
	}
	//
	grid_check_counter = config->num_objects;
	logt("schema is updated: %d grids", start, grids_stack_index);
}
