/*
 * reachability.cpp
 *
 *  Created on: Feb 25, 2021
 *      Author: teng
 */

#include "workbench.h"


bool workbench::batch_reach(reach_unit *buffer, uint num){
	if(num == 0){
		return false;
	}
	uint cur_counter = 0;
	lock();
	assert(reaches_counter+num<reaches_capacity);
	cur_counter = reaches_counter;
	reaches_counter += num;
	unlock();
	memcpy(reaches+cur_counter,buffer,sizeof(reach_unit)*num);
	return true;
}



void *reachability_unit(void *arg){
	query_context *ctx = (query_context *)arg;
	workbench *bench = (workbench *)ctx->target[0];
	Point *points = bench->points;
	reach_unit *reach_buffer = new reach_unit[2000];
	uint reach_index = 0;

	// pick one batch of point-grid pair for processing
	size_t start = 0;
	size_t end = 0;
	while(ctx->next_batch(start,end)){
		for(uint pairid=start;pairid<end;pairid++){
			uint pid = bench->grid_check[pairid].pid;
			uint gid = bench->grid_check[pairid].gid;

			uint size = bench->get_grid_size(gid);
			uint *cur_pids = bench->get_grid(gid);

			//vector<Point *> pts;
			Point *p1 = points + pid;
			for(uint i=0;i<size;i++){
				//pts.push_back(points + cur_pids[i]);
				if(pid<cur_pids[i]||!bench->grid_check[pairid].inside){
					Point *p2 = points + cur_pids[i];
					//p2->print();
					if(p1->distance(p2, true)<=ctx->config->reach_distance){
						reach_buffer[reach_index].pid1 = min(pid,cur_pids[i]);
						reach_buffer[reach_index].pid2 = max(cur_pids[i],pid);
						if(++reach_index==2000){
							bench->batch_reach(reach_buffer,reach_index);
							reach_index = 0;
						}
					}
				}
			}
		}
	}

	bench->batch_reach(reach_buffer,reach_index);
	delete []reach_buffer;
	return NULL;
}

void workbench::reachability(){

	query_context tctx;
	tctx.config = config;
	tctx.num_units = grid_check_counter;
	tctx.target[0] = (void *)this;

	// generate a new batch of reaches
	reaches_counter = 0;
	struct timeval start = get_cur_time();
	pthread_t threads[tctx.config->num_threads];

	for(int i=0;i<tctx.config->num_threads;i++){
		pthread_create(&threads[i], NULL, reachability_unit, (void *)&tctx);
	}
	for(int i = 0; i < tctx.config->num_threads; i++ ){
		void *status;
		pthread_join(threads[i], &status);
	}
	//bench->grid_check_counter = 0;
	logt("reachability compute: %d reaches are found",start,reaches_counter);
}



