/*
 * process.cpp
 *
 *  Created on: Feb 11, 2021
 *      Author: teng
 */

#include "trace.h"
#include "../index/QTree.h"



/*
 * functions for tracer
 *
 * */

void *reachability_unit(void *arg){
	query_context *ctx = (query_context *)arg;
	workbench *bench = (workbench *)ctx->target[0];
	Point *points = bench->points;
	meeting_unit *meets_buffer = new meeting_unit[200];
	uint meet_index = 0;

	size_t checked = 0;
	// pick one batch of point-grid pair for processing
	size_t start = 0;
	size_t end = 0;
	while(ctx->next_batch(start,end)){
		for(uint pairid=start;pairid<end;pairid++){
			uint pid = bench->checking_units[pairid].pid;
			uint gid = bench->checking_units[pairid].gid;
			uint offset = bench->checking_units[pairid].offset;

			uint size = min(bench->get_grid_size(gid)-offset, (uint)bench->config->zone_capacity);
			uint *cur_pids = bench->get_grid(gid)+offset;

			//vector<Point *> pts;
			Point *p1 = points + pid;
			for(uint i=0;i<size;i++){
				//pts.push_back(points + cur_pids[i]);
				Point *p2 = points + cur_pids[i];
				//p2->print();
				if(p1!=p2){
					//log("%f",dist);
					if(p1->distance(p2, true)<=ctx->config->reach_distance){
						meets_buffer[meet_index].pid1 = pid;
						meets_buffer[meet_index].pid2 = cur_pids[i];
						if(++meet_index==200){
							lock();
							assert(bench->num_meeting+meet_index<bench->meeting_capacity);
							memcpy(bench->meetings+bench->num_meeting,meets_buffer,meet_index*sizeof(meeting_unit));
							bench->num_meeting += meet_index;
							meet_index = 0;
							unlock();
						}
					}
					checked++;
				}
			}
		}
	}
	if(meet_index>0){
		lock();
		assert(bench->num_meeting+meet_index<bench->meeting_capacity);
		memcpy(bench->meetings+bench->num_meeting,meets_buffer,meet_index*sizeof(meeting_unit));
		bench->num_meeting += meet_index;
		unlock();
	}
	delete []meets_buffer;
	return NULL;
}

void reachability(workbench *bench){

	query_context tctx;
	tctx.config = bench->config;
	tctx.num_units = bench->num_checking_units;
	tctx.target[0] = (void *)bench;

	struct timeval start = get_cur_time();
	pthread_t threads[tctx.config->num_threads];
	tctx.reset();

	for(int i=0;i<tctx.config->num_threads;i++){
		pthread_create(&threads[i], NULL, reachability_unit, (void *)&tctx);
	}
	for(int i = 0; i < tctx.config->num_threads; i++ ){
		void *status;
		pthread_join(threads[i], &status);
	}
	logt("compute",start);
}

#ifdef USE_GPU
workbench *create_device_bench(workbench *bench, gpu_info *gpu);
void process_with_gpu(workbench *bench,workbench *d_bench, gpu_info *gpu);
#endif

void tracer::process(){

	bench = part->build_schema(trace, config->num_objects);
#ifdef USE_GPU
	d_bench = create_device_bench(bench, gpu);
#endif
	struct timeval start = get_cur_time();

	for(int t=0;t<config->duration;t++){
		bench->reset();
		bench->points = trace+t*config->num_objects;
		// process the objects in the packed partitions
		if(!config->gpu){
			part->partition(bench);
			for(uint start_pid=0;start_pid<config->num_objects;start_pid+=config->num_objects_per_round){
				part->lookup(bench, start_pid);
				reachability(bench);
			}
		}else{
#ifdef USE_GPU
			process_with_gpu(bench,d_bench,gpu);
#endif
		}
		logt("current round",start);

		if(config->analyze){
			bench->analyze_meetings();
		}
	}
}
