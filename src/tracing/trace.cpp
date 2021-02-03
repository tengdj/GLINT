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
	Point *points = (Point *)ctx->target[0];
	uint *offset_size = (uint *)ctx->target[1];
	int *result = (int *)ctx->target[2];
	size_t checked = 0;
	size_t reached = 0;
	while(true){
		// pick one object for generating
		int gid = ctx->fetch_one();
		if(gid<0){
			break;
		}
		int len = offset_size[2*gid+1];
		Point *cur_points = points + offset_size[2*gid];
		//n->print_node();
		if(len>2){
			for(int i=0;i<len-1;i++){
				for(int j=i+1;j<len;j++){
					double dist = cur_points[i].distance(cur_points[j], true)*1000;
					//log("%f",dist);
					result[gid] += dist<ctx->config.reach_distance;
					checked++;
				}
			}
			reached += result[gid];
		}
	}
	lock();
	ctx->checked += checked;
	ctx->found += reached;
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

void pack_grids(query_context &ctx, vector<vector<Point *>> &grids){
	struct timeval start = get_cur_time();
	size_t total_objects = 0;
	for(vector<Point *> &ps:grids){
		total_objects += ps.size();
	}
	Point *data = (Point *)new double[2*total_objects];
	uint *offset_size = new uint[grids.size()*2];
	int *result = new int[grids.size()];
	ctx.target[0] = (void *)data;
	ctx.target[1] = (void *)offset_size;
	ctx.target[2] = (void *)result;
	ctx.counter = grids.size();
	uint cur_offset = 0;
	for(int i=0;i<grids.size();i++){
		offset_size[i*2] = cur_offset;
		offset_size[i*2+1] = grids[i].size();
		for(int j=0;j<grids[i].size();j++){
			data[cur_offset++] = *grids[i][j];
		}
	}
	logt("packing data",start);
}

void process_with_gpu(query_context &ctx);

void tracer::process(){

	struct timeval start = get_cur_time();
	// test contact tracing
	size_t checked = 0;
	size_t reached = 0;
	query_context tctx;
	tctx.config = config;
	tctx.report_gap = 10;
	for(int t=0;t<config.duration;t++){
		vector<vector<Point *>> grids = part->partition(trace+t*config.num_objects, config.num_objects);
		part->clear();
		// now pack the grids assignment
		pack_grids(tctx, grids);
		// process the objects in the packed partitions
		map<size_t, int> gcount;
		for(vector<Point *> &ps:grids){
			if(gcount.find(ps.size())==gcount.end()){
				gcount[ps.size()] = 1;
			}else{
				gcount[ps.size()]++;
			}
		}
		for(auto g:gcount){
			cout<<g.first<<" "<<g.second<<endl;
		}
		process_grids(tctx);
		//process_with_gpu(tctx);

		tctx.clear();
	}
	logt("contact trace with %ld calculation use QTree %ld connected",start,tctx.checked,tctx.found);
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
	assert(config.duration*config.num_objects<=total_duration*total_num_objects);

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


