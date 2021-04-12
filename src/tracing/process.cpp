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

#ifdef USE_GPU
workbench *create_device_bench(workbench *bench, gpu_info *gpu);
void process_with_gpu(workbench *bench,workbench *d_bench, gpu_info *gpu);
#endif

void tracer::process(){

	bench = part->build_schema(trace, config->num_objects);

#ifdef USE_GPU
	if(config->gpu){
		d_bench = create_device_bench(bench, gpu);
	}
#endif
	struct timeval start = get_cur_time();

	for(int t=0;t<config->duration;t++){
		log("");
		bench->reset();
		bench->points = trace+t*config->num_objects;
		bench->cur_time = t;
		// process the coordinate in this time point
		if(!config->gpu){
			struct timeval ct = get_cur_time();
			bench->filter();
			bench->pro.filter_time += get_time_elapsed(ct,true);
			bench->reachability();
			bench->pro.refine_time += get_time_elapsed(ct,true);
			bench->update_meetings();
			bench->pro.meeting_update_time += get_time_elapsed(ct,true);
		}else{
#ifdef USE_GPU
			process_with_gpu(bench,d_bench,gpu);
#endif
		}
		if(bench->meeting_counter>0&&t==config->duration-1){
			int luck = get_rand_number(bench->meeting_counter);
			print_trace(bench->meetings[luck].pid1);
			print_trace(bench->meetings[luck].pid2);
		}
		if(config->analyze_grid){
			bench->analyze_grids();
		}
		if(config->analyze_reach){
			bench->analyze_reaches();
		}
		if(config->analyze_meeting){
			bench->analyze_meeting_buckets();
		}
		if(config->dynamic_schema&&!config->gpu){
			struct timeval ct = get_cur_time();
			bench->update_schema();
			bench->pro.index_update_time += get_time_elapsed(ct,true);
		}
		logt("round %d",start,t+config->start_time);
		bench->current_bucket = !bench->current_bucket;
		bench->pro.rounds++;
	}

	bench->print_profile();
}
