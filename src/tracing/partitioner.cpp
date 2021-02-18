/*
 * partitioner.cpp
 *
 *  Created on: Feb 1, 2021
 *      Author: teng
 */

#include "trace.h"

workbench *grid_partitioner::build_schema(Point *objects, size_t num_objects){
	if(grid){
		delete grid;
	}
	grid = new Grid(mbr, config.grid_width);
	size_t num_grids = grid->get_grid_num();
	workbench *bench = new workbench(config);
	bench->claim_space(num_grids);
	return bench;
}

void grid_partitioner::partition(workbench *bench){
	// the schema is built
	assert(bench);
	struct timeval start = get_cur_time();

	double x_buffer = config.reach_distance*degree_per_meter_longitude(grid->space.low[1]);
	double y_buffer = config.reach_distance*degree_per_meter_latitude;

	// assign each object to proper grids
	for(int pid=0;pid<bench->config.num_objects;pid++){
		Point *p = bench->points+pid;
		size_t gid = grid->getgridid(p);
		bench->insert(gid,pid);
	}
	logt("partition %ld objects into %ld grids",start,bench->config.num_objects, bench->num_grids);

	// the query process
	// each point is associated with a list of grids
	for(size_t pid=0;pid<bench->config.num_objects;pid++){
		size_t gid = grid->getgridid(bench->points+pid);
		bench->check(gid,pid);

		size_t gid_code = grid->border_grids(bench->points+pid,x_buffer,y_buffer);
		bool left = gid_code & 8;
		bool right = gid_code & 4;
		bool top = gid_code &2;
		bool bottom = gid_code &1;
		if(bottom&&left){
			bench->check(gid-grid->dimx-1,pid);
		}
		if(bottom){
			bench->check(gid-grid->dimx,pid);
		}
		if(bottom&&right){
			bench->check(gid-grid->dimx+1,pid);
		}
		if(left){
			bench->check(gid-1,pid);
		}
		if(right){
			bench->check(gid+1,pid);
		}
		if(top&&left){
			bench->check(gid+grid->dimx-1, pid);
		}
		if(top){
			bench->check(gid+grid->dimx,pid);
		}
		if(top&&right){
			bench->check(gid+grid->dimx+1, pid);
		}
	}

	logt("query",start);
}


workbench *qtree_partitioner::build_schema(Point *points, size_t num_objects){
	struct timeval start = get_cur_time();
	QTConfig qconfig;
	qconfig.min_width = config.reach_distance;
	qconfig.max_objects = config.grid_capacity;
	config.x_buffer = config.reach_distance*degree_per_meter_longitude(mbr.low[1]);
	config.y_buffer = config.reach_distance*degree_per_meter_latitude;
	qconfig.x_buffer = config.x_buffer;
	qconfig.y_buffer = config.y_buffer;
	QTNode *qtree = new QTNode(mbr, &qconfig);
	qconfig.points = points;

	for(uint pid=0;pid<num_objects;pid++){
		//log("%d",pid);
		qtree->insert(pid);
	}
	// set the ids and other stuff
	qtree->finalize();
	size_t num_grids = qtree->leaf_count();


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
				// is leaf
				if((bench->schema[curoff].children[loc]&1)){
					gid = bench->schema[curoff].children[loc]>>1;
					break;
				}else{
					curoff = bench->schema[curoff].children[loc]>>1;
				}
			}
			bench->insert(gid, pid);
		}
	}
	return NULL;
}

void lookup(QTSchema *schema, Point *p, uint curoff, vector<uint> &gids, double x_buffer, double y_buffer){

	// could be possibly in multiple children with buffers enabled
	bool top = (p->y>schema[curoff].mid_y-y_buffer);
	bool bottom = (p->y<=schema[curoff].mid_y+y_buffer);
	bool left = (p->x<=schema[curoff].mid_x+x_buffer);
	bool right = (p->x>schema[curoff].mid_x-x_buffer);
	bool need_check[4] = {bottom&&left, bottom&&right, top&&left, top&&right};
	for(int i=0;i<4;i++){
		if(need_check[i]){
			if((schema[curoff].children[i]&1)){
				gids.push_back(schema[curoff].children[i]>>1);
			}else{
				lookup(schema, p, schema[curoff].children[i]>>1, gids, x_buffer, y_buffer);
			}
		}
	}
}

// single thread function for looking up the schema to generate point-grid pairs for processing
void *lookup_unit(void *arg){
	query_context *qctx = (query_context *)arg;
	workbench *bench = (workbench *)qctx->target[0];
	QTSchema *schema = bench->schema;

	// pick one batch of point-grid pair for processing
	size_t start = 0;
	size_t end = 0;
	vector<uint> gids;
	checking_unit *cubuffer = (checking_unit *)malloc(sizeof(checking_unit)*200);
	uint buffer_index = 0;
	while(qctx->next_batch(start,end)){
		for(uint pid=start;pid<end;pid++){
			Point *p = bench->points+pid;
			lookup(schema, p, 0, gids, qctx->config.x_buffer, qctx->config.y_buffer);

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
					offset += bench->config.zone_capacity;
				}
			}
			gids.clear();
		}
	}
	bench->batch_check(cubuffer, buffer_index);
	return NULL;
}



void qtree_partitioner::partition(workbench *bench){
	// the schema has to be built
	assert(bench);
	struct timeval start = get_cur_time();

	// partitioning current batch of objects with the existing schema
	pthread_t threads[config.num_threads];
	query_context qctx;
	qctx.config = config;
	qctx.num_units = bench->config.num_objects;
	qctx.target[0] = (void *)bench;

	for(int i=0;i<config.num_threads;i++){
		pthread_create(&threads[i], NULL, partition_unit, (void *)&qctx);
	}
	for(int i = 0; i < config.num_threads; i++ ){
		void *status;
		pthread_join(threads[i], &status);
	}
	logt("partition",start);

	// tree lookups
	qctx.reset();
	for(int i=0;i<config.num_threads;i++){
		pthread_create(&threads[i], NULL, lookup_unit, (void *)&qctx);
	}
	for(int i = 0; i < config.num_threads; i++ ){
		void *status;
		pthread_join(threads[i], &status);
	}
	logt("lookup",start);
}
