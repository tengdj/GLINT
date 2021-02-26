/*
 * meeting.cpp
 *
 *  Created on: Feb 25, 2021
 *      Author: teng
 */


#include "workbench.h"



/*
 *
 * update the meetings maintained with reachability information collected in this round
 *
 * */


bool workbench::batch_meet(meeting_unit *buffer, uint num){
	if(num == 0){
		return false;
	}
	uint cur_counter = 0;
	assert(meeting_counter+num<meeting_capacity);
	lock();
	cur_counter = meeting_counter;
	meeting_counter += num;
	unlock();
	memcpy(meetings+cur_counter,buffer,sizeof(meeting_unit)*num);
	return true;
}



void *update_meetings_unit(void *arg){
	query_context *ctx = (query_context *)arg;
	workbench *bench = (workbench *)ctx->target[0];

	size_t start = 0;
	size_t end = 0;
	while(ctx->next_batch(start,end)){
		for(uint rid=start;rid<end;rid++){
			uint pid1 = bench->reaches[rid].pid1;
			uint pid2 = bench->reaches[rid].pid2;
			uint bid = (pid1+pid2)%bench->config->num_meeting_buckets;
			meeting_unit *bucket = bench->meeting_buckets+bid*bench->meeting_bucket_capacity;
			bool updated = false;
			for(uint i=0;i<bench->meeting_buckets_counter_tmp[bid];i++){
				// a former meeting is encountered, update it
				if(bucket[i].pid1==pid1&&bucket[i].pid2==pid2){
					bucket[i].end = bench->cur_time;
					updated = true;
					break;
				}
			}
			if(!updated){
				uint loc = 0;
				bench->lock(bid);
				loc = bench->meeting_buckets_counter[bid]++;
				bench->unlock(bid);
				assert(loc<bench->meeting_bucket_capacity);
				bucket[loc].pid1 = pid1;
				bucket[loc].pid2 = pid2;
				bucket[loc].start = bench->cur_time;
				bucket[loc].end = bench->cur_time;
			}
		}
	}
	return NULL;
}

void workbench::update_meetings(){
	struct timeval start = get_cur_time();

	query_context tctx;
	tctx.config = config;
	tctx.target[0] = (void *)this;
	tctx.num_units = reaches_counter;
	pthread_t threads[tctx.config->num_threads];

	for(int i=0;i<tctx.config->num_threads;i++){
		pthread_create(&threads[i], NULL, update_meetings_unit, (void *)&tctx);
	}
	for(int i = 0; i < tctx.config->num_threads; i++ ){
		void *status;
		pthread_join(threads[i], &status);
	}
	int mc = 0;
	for(int i=0;i<config->num_meeting_buckets;i++){
		mc += meeting_buckets_counter[i];
	}
	logt("update meeting: %d meetings active",start,mc);
}



void *compact_meetings_unit(void *arg){
	query_context *ctx = (query_context *)arg;
	workbench *bench = (workbench *)ctx->target[0];

	size_t start = 0;
	size_t end = 0;
	meeting_unit *mu_buffer = new meeting_unit[200];
	uint mu_index = 0;

	while(ctx->next_batch(start,end)){
		for(uint bid=start;bid<end;bid++){
			meeting_unit *bucket = bench->meeting_buckets+bid*bench->meeting_bucket_capacity;
			int front_idx = 0;
			int back_idx = bench->meeting_buckets_counter[bid]-1;
			int active_count = 0;
			for(;front_idx<=back_idx;front_idx++){
				// this meeting is over
				if(bucket[front_idx].end<bench->cur_time){
					// dump to valid list and copy one from the back end
					if(bucket[front_idx].end-bucket[front_idx].start>=bench->config->min_meet_time){
						mu_buffer[mu_index++] = bucket[front_idx];
						if(mu_index==200){
							bench->batch_meet(mu_buffer,mu_index);
							mu_index = 0;
						}
					}
					for(;back_idx>front_idx;back_idx--){
						if(bucket[back_idx].end==bench->cur_time){
							break;
							// dump to valid list if needed or disregarded
						}else if(bucket[back_idx].end-bucket[back_idx].start>=bench->config->min_meet_time){
							mu_buffer[mu_index++] = bucket[back_idx];
							if(mu_index==200){
								bench->batch_meet(mu_buffer,mu_index);
								mu_index = 0;
							}
						}
					}
					if(front_idx<back_idx){
						bucket[front_idx] = bucket[back_idx];
						active_count++;
						back_idx--;
					}
				}else{
					active_count++;
				}
			}

			bench->meeting_buckets_counter[bid] = active_count;
			bench->meeting_buckets_counter_tmp[bid] = active_count;
		}
	}
	bench->batch_meet(mu_buffer,mu_index);
	delete []mu_buffer;
	return NULL;
}

void workbench::compact_meetings(){
	struct timeval start = get_cur_time();
	query_context tctx;
	tctx.config = config;
	tctx.target[0] = (void *)this;
	tctx.num_units = config->num_meeting_buckets;
	pthread_t threads[tctx.config->num_threads];

	for(int i=0;i<tctx.config->num_threads;i++){
		pthread_create(&threads[i], NULL, compact_meetings_unit, (void *)&tctx);
	}
	for(int i = 0; i < tctx.config->num_threads; i++ ){
		void *status;
		pthread_join(threads[i], &status);
	}
	logt("compact meeting: %d meetings recorded",start,meeting_counter);

}

