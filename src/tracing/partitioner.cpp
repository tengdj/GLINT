/*
 * partitioner.cpp
 *
 *  Created on: Feb 1, 2021
 *      Author: teng
 */

#include "trace.h"


void *grid_partition(void *arg){
	query_context *ctx = (query_context *)arg;
	Grid *grid = (Grid *)ctx->target[0];
	Point *points = (Point *)ctx->target[1];
	vector<vector<Point *>> *grids = (vector<vector<Point *>> *)ctx->target[2];

	double x_buffer = ctx->config.reach_distance/1000*degree_per_kilometer_longitude(grid->space.low[1]);
	double y_buffer = ctx->config.reach_distance/1000*degree_per_kilometer_latitude;
	while(true){
		// pick one object for generating
		int pid = ctx->fetch_one();
		if(pid<0){
			break;
		}
		Point *p = points+pid;
		vector<int> gids= grid->getgrids(p,x_buffer,y_buffer);
		for(int gid:gids){
			ctx->lock(gid);
			(*grids)[gid].push_back(p);
			ctx->unlock(gid);
		}
		gids.clear();
	}
	return NULL;
}

vector<vector<Point *>> grid_partitioner::partition(Point *points, size_t num_objects){
	struct timeval start = get_cur_time();
	double x_buffer = config.reach_distance/1000*degree_per_kilometer_longitude(mbr.low[1]);
	double y_buffer = config.reach_distance/1000*degree_per_kilometer_latitude;

	query_context tctx;

	tctx.target[0] = (void *)grid;
	tctx.target[1] = (void *)points;
	tctx.target[2] = (void *)&grids;

	tctx.counter = num_objects;
	pthread_t threads[config.num_threads];
	for(int i=0;i<config.num_threads;i++){
		pthread_create(&threads[i], NULL, grid_partition, (void *)&tctx);
	}
	for(int i = 0; i < config.num_threads; i++ ){
		void *status;
		pthread_join(threads[i], &status);
	}
	logt("space is partitioned into %d grids",start,grid->get_grid_num());
	return grids;
}

void *split_qtree_unit(void *arg){
	query_context *ctx = (query_context *)arg;
	queue<QTNode *> *qq = (queue<QTNode *> *)ctx->target[0];
	while(!qq->empty()||ctx->counter>0){
		QTNode *node = NULL;
		ctx->lock();
		if(!qq->empty()){
			node = qq->front();
			qq->pop();
		}
		ctx->unlock();
		if(node){
			ctx->busy();
			if(node->split()){
				ctx->lock();
				qq->push(node->children[0]);
				qq->push(node->children[1]);
				qq->push(node->children[2]);
				qq->push(node->children[3]);
				ctx->unlock();
			}
			ctx->idle();
		}

		if(qq->empty()){
			sleep(1);
		}
	}
	return NULL;
}

void split_qtree(QTNode *qtree, Point *points, size_t num_objects, int num_threads){
	query_context tctx;
	queue<QTNode *> qq;
	qq.push(qtree);
	tctx.target[0] = (void *)&qq;
	pthread_t threads[num_threads];
	for(int i=0;i<num_threads;i++){
		pthread_create(&threads[i], NULL, split_qtree_unit, (void *)&tctx);
	}
	for(int i = 0; i < num_threads; i++ ){
		void *status;
		pthread_join(threads[i], &status);
	}
}

vector<vector<Point *>> qtree_partitioner::partition(Point *points, size_t num_objects){
	struct timeval start = get_cur_time();
	grids.clear();
	assert(qtree==NULL);
	qtree = new QTNode(mbr, &qconfig);
	qtree->objects.resize(num_objects);
	for(int i=0;i<num_objects;i++){
		qtree->objects[i] = (points+i);
	}
	split_qtree(qtree, points, num_objects,config.num_threads);
	qtree->get_leafs(grids, true);
	logt("space is partitioned into %d grids %ld objects",start,qtree->leaf_count(),qtree->num_objects());
	qtree->print();
	return grids;
}


void qtree_partitioner::clear(){
	for(vector<Point *> &ps:grids){
		ps.clear();
	}
	grids.clear();
	delete qtree;
	qtree = NULL;
}

