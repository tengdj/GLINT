/*
 * partitioner.cpp
 *
 *  Created on: Feb 1, 2021
 *      Author: teng
 */

#include "trace.h"

workbench *qtree_partitioner::build_schema(Point *points, size_t num_objects){
	struct timeval start = get_cur_time();
	config->x_buffer = config->reach_distance*degree_per_meter_longitude(mbr.low[1]);
	config->y_buffer = config->reach_distance*degree_per_meter_latitude;
	QTNode *qtree = new QTNode(mbr, config, points);

	for(uint pid=0;pid<num_objects;pid++){
		//log("%d",pid);
		qtree->insert(pid);
	}
	// set the ids and other stuff
	qtree->finalize();
	size_t num_grids = qtree->leaf_count();
	//qtree->print();

	workbench *bench = new workbench(config);
	bench->claim_space(num_grids);

	bench->num_nodes = qtree->node_count();
	bench->schema = qtree->create_schema();

	delete qtree;
	logt("partitioning schema is with %d grids",start,num_grids);
	return bench;
}

// single thread function for assigning objects into grids following certain schema
void *partition_unit(void *arg){
	query_context *qctx = (query_context *)arg;
	workbench *bench = (workbench *)qctx->target[0];
	QTSchema *schema = bench->schema;

	// pick one batch of point-grid pair for processing
	size_t start = 0;
	size_t end = 0;
	while(qctx->next_batch(start,end)){
		for(uint pid=start;pid<end;pid++){
			uint gid = 0;
			uint curoff = 0;
			Point *p = bench->points+pid;
			while(true){
				int loc = (p->y>bench->schema[curoff].mid_y)*2+(p->x>bench->schema[curoff].mid_x);
				uint child_offset = bench->schema[curoff].children[loc];
				// is leaf
				if(bench->schema[child_offset].isleaf){
					gid = bench->schema[child_offset].node_id;
					break;
				}
				curoff = child_offset;
			}
			bench->insert(gid, pid);
		}
	}
	return NULL;
}



void qtree_partitioner::partition(workbench *bench){
	// the schema has to be built
	assert(bench);
	struct timeval start = get_cur_time();

	// partitioning current batch of objects with the existing schema
	pthread_t threads[config->num_threads];
	query_context qctx;
	qctx.config = config;
	qctx.num_units = bench->config->num_objects;
	qctx.target[0] = (void *)bench;

	for(int i=0;i<config->num_threads;i++){
		pthread_create(&threads[i], NULL, partition_unit, (void *)&qctx);
	}
	for(int i = 0; i < config->num_threads; i++ ){
		void *status;
		pthread_join(threads[i], &status);
	}
	logt("partition",start);
}


/*
 *
 * the CPU functions for looking up QTree with points and generate pid-gid-offset
 *
 * */


void lookup(QTSchema *schema, Point *p, uint curoff, vector<uint> &gids, double x_buffer, double y_buffer){

	// could be possibly in multiple children with buffers enabled
	bool top = (p->y>schema[curoff].mid_y-y_buffer);
	bool bottom = (p->y<=schema[curoff].mid_y+y_buffer);
	bool left = (p->x<=schema[curoff].mid_x+x_buffer);
	bool right = (p->x>schema[curoff].mid_x-x_buffer);
	bool need_check[4] = {bottom&&left, bottom&&right, top&&left, top&&right};
	for(int i=0;i<4;i++){
		if(need_check[i]){
			uint child_offset = schema[curoff].children[i];
			if(schema[child_offset].isleaf){
				gids.push_back(schema[child_offset].node_id);
			}else{
				lookup(schema, p, child_offset, gids, x_buffer, y_buffer);
			}
		}
	}
}

int one = 0;
int mone = 0;
// single thread function for looking up the schema to generate point-grid pairs for processing
void *lookup_unit(void *arg){
	query_context *qctx = (query_context *)arg;
	workbench *bench = (workbench *)qctx->target[0];
	QTSchema *schema = bench->schema;

	// pick one point for looking up
	size_t start = 0;
	size_t end = 0;
	vector<uint> gids;
	checking_unit *cubuffer = new checking_unit[200];
	uint buffer_index = 0;
	while(qctx->next_batch(start,end)){
		for(uint pid=start;pid<end;pid++){
			Point *p = bench->points+pid;
			lookup(schema, p, 0, gids, qctx->config->x_buffer, qctx->config->y_buffer);
			lock();
			if(gids.size()==1){
				one++;
			}
			if(gids.size()>1){
				mone++;
			}
			unlock();
			for(uint gid:gids){
				assert(gid<bench->num_grids);
				uint offset = 0;
				while(offset<bench->get_grid_size(gid)){

					cubuffer[buffer_index].pid = pid;
					cubuffer[buffer_index].gid = gid;
					cubuffer[buffer_index].offset = offset;
					buffer_index++;
					if(buffer_index==200){
						bench->batch_check(cubuffer, buffer_index);
						buffer_index = 0;
					}
					offset += bench->config->zone_capacity;
				}
			}
			gids.clear();
		}
	}
	bench->batch_check(cubuffer, buffer_index);
	log("%d %d",one,mone);
	delete []cubuffer;
	return NULL;
}

void qtree_partitioner::lookup(workbench *bench, uint start_pid){
	// the schema has to be built
	assert(bench);
	struct timeval start = get_cur_time();
	// reset the units
	bench->num_unit_lookup = 0;
	// partitioning current batch of objects with the existing schema
	pthread_t threads[config->num_threads];
	query_context qctx;
	qctx.report_gap = 100;
	qctx.config = config;
	qctx.counter = start_pid;
	qctx.num_units = min(bench->config->num_objects, start_pid+config->num_objects_per_round);
	qctx.target[0] = (void *)bench;

	// tree lookups
	for(int i=0;i<config->num_threads;i++){
		pthread_create(&threads[i], NULL, lookup_unit, (void *)&qctx);
	}
	for(int i = 0; i < config->num_threads; i++ ){
		void *status;
		pthread_join(threads[i], &status);
	}
	logt("lookup",start);
}


