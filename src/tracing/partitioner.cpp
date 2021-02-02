/*
 * partitioner.cpp
 *
 *  Created on: Feb 1, 2021
 *      Author: teng
 */

#include "trace.h"



void grid_partitioner::index(Point *p, size_t num_objects){
	struct timeval start = get_cur_time();
	grid = new Grid(mbr, config.grid_width);
	grids.resize(grid->get_grid_num()+1);
	logt("space is partitioned into %d grids",start,grid->get_grid_num());
}

void grid_partitioner::partition(Point *points, size_t num_objects){
	struct timeval start = get_cur_time();
	double x_buffer = config.reach_distance/1000*degree_per_kilometer_longitude(mbr.low[1]);
	double y_buffer = config.reach_distance/1000*degree_per_kilometer_latitude;
	for(int o=0;o<num_objects;o++){
		Point *p = points+o;
		vector<int> gids= grid->getgrids(p,x_buffer,y_buffer);
		for(int gid:gids){
			grids[gid].push_back(p);
		}
		gids.clear();
	}
	logt("space is partitioned into %d grids",start,grid->get_grid_num());
}

void *insert_qtree_unit(void *arg){
	query_context *ctx = (query_context *)arg;
	QTNode *qtree = (QTNode *)ctx->target[0];
	Point *points = (Point *)ctx->target[1];
	while(true){
		// pick one object for generating
		int pid = ctx->fetch_one();
		if(pid<0){
			break;
		}
		qtree->insert(points+pid);
	}
	return NULL;
}

void insert_qtree(QTNode *qtree, Point *points, size_t num_objects, int num_threads){
	query_context tctx;
	tctx.target[0] = (void *)qtree;
	tctx.target[1] = (void *)points;
	tctx.counter = num_objects;
	pthread_t threads[num_threads];
	for(int i=0;i<num_threads;i++){
		pthread_create(&threads[i], NULL, insert_qtree_unit, (void *)&tctx);
	}
	for(int i = 0; i < num_threads; i++ ){
		void *status;
		pthread_join(threads[i], &status);
	}
}

void qtree_partitioner::index(Point *points, size_t num_objects){
	struct timeval start = get_cur_time();
	qconfig.grid_width = config.grid_width;
	qconfig.max_objects = config.max_objects_per_grid;
	qconfig.x_buffer = config.reach_distance*degree_per_kilometer_longitude(mbr.low[1])/1000;
	qconfig.y_buffer = config.reach_distance*degree_per_kilometer_latitude/1000;
	//printf("%f %f %f %f\n",qconfig.x_buffer, qconfig.y_buffer, qconfig.x_buffer/degree_per_kilometer_longitude(mbr.low[1]),qconfig.y_buffer/degree_per_kilometer_latitude);
	qtree = new QTNode(mbr);
	qtree->set_config(&qconfig);
	insert_qtree(qtree, points, num_objects, 1);
	size_t num_obj = qtree->num_objects();
	qtree->fix_structure();
	logt("building qtree with %ld points with %d max_objects %d leafs", start, num_obj, config.max_objects_per_grid, qconfig.num_leafs);
}


void qtree_partitioner::partition(Point *points, size_t num_objects){
	assert(qtree);
	struct timeval start = get_cur_time();
	insert_qtree(qtree, points, num_objects, config.num_threads);
	qtree->get_leafs(grids, true);
	logt("space is partitioned into %d grids",start,qtree->leaf_count());
}


void qtree_partitioner::clear(){
	for(vector<Point *> &ps:grids){
		ps.clear();
	}
	qtree->fix_structure();
}

