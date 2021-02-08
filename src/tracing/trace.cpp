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
	uint *pids = (uint *)ctx->target[1];
	offset_size *os = (offset_size *)ctx->target[2];
	uint *result = (uint *)ctx->target[3];
	size_t checked = 0;
	size_t reached = 0;
	int max_len = 0;
	while(true){
		// pick one object for generating
		size_t start = 0;
		size_t end = 0;
		if(!ctx->next_batch(start,end)){
			break;
		}
		for(int gid=start;gid<end;gid++){
			uint len = os[gid].size;
			uint *cur_pids = pids + os[gid].offset;
			result[gid] = 0;
//			if(max_len<len){
//				log("%d %d",gid,len);
//				max_len = len;
//			}
			//log("%d %d",gid,len);
//			if(len>1000){
//				len=1000;
//				invalid++;
//			}
			//vector<Point *> pts;
			if(len>2){
				for(uint i=0;i<len-1;i++){
					//pts.push_back(points + cur_pids[i]);
					for(uint j=i+1;j<len;j++){
						Point *p1 = points + cur_pids[i];
						Point *p2 = points + cur_pids[j];
						double dist = p1->distance(*p2, true)*1000;
						//log("%f",dist);
						result[gid] += dist<ctx->config.reach_distance;
						checked++;
					}
				}
				reached += result[gid];
//				if(len>100){
//					lock();
//					print_points(pts);
//					unlock();
//					exit(0);
//				}

			}
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
	tctx.clear();

	for(int i=0;i<tctx.config.num_threads;i++){
		pthread_create(&threads[i], NULL, process_grid_unit, (void *)&tctx);
	}
	for(int i = 0; i < tctx.config.num_threads; i++ ){
		void *status;
		pthread_join(threads[i], &status);
	}
	logt("compute",start);
}


void process_with_gpu(query_context &ctx);

void tracer::process(){

	struct timeval start = get_cur_time();
	// test contact tracing
	size_t checked = 0;
	size_t reached = 0;

	// first level of partition
	uint *pids = new uint[config.num_objects];
	for(int i=0;i<config.num_objects;i++){
		pids[i] = i;
	}

	for(int t=0;t<config.duration;t++){
		query_context tctx = part->partition(trace+t*config.num_objects, pids, config.num_objects);
		// process the objects in the packed partitions
		if(!config.gpu){
			process_grids(tctx);
		}else{
			process_with_gpu(tctx);
		}
		checked += tctx.checked;
		reached += tctx.found;
	}
	logt("contact trace with %ld calculation %ld connected",start,checked,reached);
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


