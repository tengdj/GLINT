/*
 * generator.cpp
 *
 *  Created on: Jan 19, 2021
 *      Author: teng
 */

#include "trace.h"

#include "../index/QTree.h"






/*
 * functions for tracer
 *
 * */

void *process_grid_unit(void *arg){
	query_context *ctx = (query_context *)arg;
	vector<vector<Point *>> *grids = (vector<vector<Point *>> *)ctx->target[0];
	size_t checked = 0;
	size_t reached = 0;
	while(true){
		// pick one object for generating
		int gid = ctx->fetch_one();
		if(gid<0){
			break;
		}
		int len = (*grids)[gid].size();
		//n->print_node();
		if(len>2){
			for(int i=0;i<len-1;i++){
				for(int j=i+1;j<len;j++){
					double dist = (*grids)[gid][i]->distance(*(*grids)[gid][j], true)*1000;
					//log("%f",dist);
					if(dist<ctx->config.reach_distance){
						reached++;
					}
					checked++;
				}
			}
		}
	}
	lock();
	*(size_t *)ctx->target[1] += checked;
	*(size_t *)ctx->target[2] += reached;
	unlock();
	return NULL;
}

void process_grids(query_context &tctx){
	struct timeval start = get_cur_time();
	pthread_t threads[tctx.config.num_threads];

	for(int i=0;i<tctx.config.num_threads;i++){
		pthread_create(&threads[i], NULL, process_grid_unit, (void *)&tctx);
	}
	for(int i = 0; i < tctx.config.num_threads; i++ ){
		void *status;
		pthread_join(threads[i], &status);
	}
	logt("compute",start);
}

void tracer::process(){

	struct timeval start = get_cur_time();
	// test contact tracing
	size_t checked = 0;
	size_t reached = 0;
	for(int t=0;t<config.duration;t++){
		vector<vector<Point *>> grids = part->partition(trace+t*config.num_objects, config.num_objects);
		query_context tctx;
		tctx.config = config;
		tctx.target[0] = (void *)&grids;
		tctx.counter = grids.size();
		tctx.target[1] = (void *)&checked;
		tctx.target[2] = (void *)&reached;
		process_grids(tctx);

		part->clear();
	}
	logt("contact trace with %ld calculation use QTree %ld connected",start,checked,reached);
}

void tracer::dumpTo(const char *path) {
	struct timeval start_time = get_cur_time();
	ofstream wf(path, ios::out|ios::binary|ios::trunc);
	wf.write((char *)&config.num_objects, sizeof(config.num_objects));
	wf.write((char *)&config.duration, sizeof(config.duration));
	wf.write((char *)&mbr, sizeof(mbr));
	size_t num_points = config.duration*config.num_objects;
	wf.write((char *)trace, sizeof(Point)*num_points);
	wf.close();
	logt("dumped to %s",start_time,path);
}

void tracer::loadFrom(const char *path) {

	int total_num_objects;
	int total_duration;
	struct timeval start_time = get_cur_time();
	ifstream in(path, ios::in | ios::binary);
	in.read((char *)&total_num_objects, sizeof(total_num_objects));
	in.read((char *)&total_duration, sizeof(total_duration));
	in.read((char *)&mbr, sizeof(mbr));
	mbr.to_squre(true);
	assert(config.duration<=total_duration);
	assert(config.num_objects<=total_num_objects);

	trace = (Point *)malloc(config.duration*config.num_objects*sizeof(Point));
	for(int i=0;i<config.duration;i++){
		in.read((char *)(trace+i*config.num_objects), config.num_objects*sizeof(Point));
		if(total_num_objects>config.num_objects){
			in.seekg((total_num_objects-config.num_objects)*sizeof(Point), ios_base::cur);
		}
	}

	in.close();
	logt("loaded %d objects last for %d seconds from %s",start_time, config.num_objects, config.duration, path);
	owned_trace = true;
}

void tracer::print_trace(double sample_rate){
	print_points(trace,config.num_objects,sample_rate);
}


