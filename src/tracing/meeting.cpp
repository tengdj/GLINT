/*
 * meeting.cpp
 *
 *  Created on: Feb 25, 2021
 *      Author: teng
 */


#include "workbench.h"



bool workbench::batch_meet(meeting_unit *buffer, uint num){
	if(num == 0){
		return false;
	}
	uint cur_counter = 0;
	lock();
	cur_counter = meeting_counter;
	meeting_counter += num;
	unlock();
	if(meeting_counter<meeting_capacity){
		memcpy(meetings+cur_counter,buffer,sizeof(meeting_unit)*num);
	}
	return true;
}

/*
 *
 * update the meetings maintained with reachability information collected in this round
 *
 * */

void *update_meetings_unit(void *arg){
	query_context *ctx = (query_context *)arg;
	workbench *bench = (workbench *)ctx->target[0];

	meeting_unit *meetings = new meeting_unit[200];
	uint meeting_index = 0;

	size_t start = 0;
	size_t end = 0;
	while(ctx->next_batch(start,end)){
		for(uint bid=start;bid<end;bid++){

			meeting_unit *bucket_new = bench->meeting_buckets[bench->current_bucket]+bid*bench->config->bucket_size;
			meeting_unit *bucket_old = bench->meeting_buckets[!bench->current_bucket]+bid*bench->config->bucket_size;

			uint size_new = bench->meeting_buckets_counter[bench->current_bucket][bid];
			uint size_old = bench->meeting_buckets_counter[!bench->current_bucket][bid];

			for(uint i=0;i<size_old&&i<bench->config->bucket_size;i++){
				bool updated = false;
				for(uint j=0;j<size_new&&j<bench->config->bucket_size;j++){
					if(bucket_new[i].pid1==bucket_old[j].pid1&&
					   bucket_new[i].pid2==bucket_old[j].pid2){
						bucket_new[i].start = bucket_old[i].start;
						updated = true;
						break;
					}
				}
				// the old meeting is over
				if(!updated&&
					bench->cur_time-bucket_old[i].start>=bench->config->min_meet_time){
					meetings[meeting_index++] = bucket_old[i];
					if(meeting_index==200){
						bench->batch_meet(meetings, 200);
						meeting_index = 0;
					}
				}
			}
			bench->meeting_buckets_counter[!bench->current_bucket][bid] = 0;
		}
	}
	bench->batch_meet(meetings, meeting_index);
	return NULL;
}

void workbench::update_meetings(){
	struct timeval start = get_cur_time();

	query_context tctx;
	tctx.config = config;
	tctx.target[0] = (void *)this;
	tctx.num_units = config->num_meeting_buckets;
	pthread_t threads[tctx.config->num_threads];

	meeting_counter = 0;
	for(int i=0;i<tctx.config->num_threads;i++){
		pthread_create(&threads[i], NULL, update_meetings_unit, (void *)&tctx);
	}
	for(int i = 0; i < tctx.config->num_threads; i++ ){
		void *status;
		pthread_join(threads[i], &status);
	}
	int mc = 0;
	for(int i=0;i<config->num_meeting_buckets;i++){
		mc += meeting_buckets_counter[current_bucket][i];
	}
	logt("update meeting: %d meetings active, %d meeting identified",start,mc,meeting_counter);
}

