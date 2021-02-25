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
		bench->reset();
		bench->points = trace+t*config->num_objects;
		bench->cur_time = t;
		// process the coordinate in this time points
		if(!config->gpu){
			bench->partition();
			bench->lookup();
			bench->reachability();
			bench->update_meetings();
			bench->compact_meetings();
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
		if(config->analyze){
			bench->analyze_grids();
			//bench->analyze_meetings();
		}
		logt("round %d",start,t);
		log("");
	}
}
