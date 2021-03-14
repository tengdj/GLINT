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

void *update_meetings_unit(void *arg){
	query_context *ctx = (query_context *)arg;
	workbench *bench = (workbench *)ctx->target[0];

	size_t start = 0;
	size_t end = 0;
	while(ctx->next_batch(start,end)){
		for(uint bid=start;bid<end;bid++){

			meeting_unit *bucket_new = bench->meeting_buckets[bench->current_bucket]+bid*bench->meeting_bucket_capacity;
			meeting_unit *bucket_old = bench->meeting_buckets[!bench->current_bucket]+bid*bench->meeting_bucket_capacity;

			uint size_new = bench->meeting_buckets_counter[bench->current_bucket][bid];
			uint size_old = bench->meeting_buckets_counter[!bench->current_bucket][bid];

			for(uint i=0;i<size_old&&i<bench->meeting_bucket_capacity;i++){
				bool updated = false;
				for(uint j=0;j<size_new&&j<bench->meeting_bucket_capacity;j++){
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
					lock();
					uint meeting_idx = bench->meeting_counter++;
					unlock();
					assert(meeting_idx<bench->meeting_capacity);
					bench->meetings[meeting_idx] = bucket_old[i];
				}
			}
			bench->meeting_buckets_counter[!bench->current_bucket][bid] = 0;
		}
	}
	return NULL;
}

void workbench::update_meetings(){
	struct timeval start = get_cur_time();

	query_context tctx;
	tctx.config = config;
	tctx.target[0] = (void *)this;
	tctx.num_units = config->num_meeting_buckets;
	pthread_t threads[tctx.config->num_threads];

	uint old_counter = this->meeting_counter;
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
	logt("update meeting: %d meetings active, %d meeting identified",start,mc,meeting_counter - old_counter);
}

