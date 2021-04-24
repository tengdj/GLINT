/*
 * meeting.cpp
 *
 *  Created on: Feb 25, 2021
 *      Author: teng
 */


#include "workbench.h"

/*
 * sort-based method
 * */

inline int partition(meeting_unit *arr, int low, int high){
    int pivot = arr[high].pid1;    // pivot
    int i = (low - 1);  // Index of smaller element

    meeting_unit tmp;
    for (int j = low; j <= high- 1; j++){
        // If current element is smaller than or
        // equal to pivot
        if (arr[j].pid1 <= pivot){
            i++;    // increment index of smaller element
            tmp = arr[i];
            arr[i] = arr[j];
            arr[j] = tmp;
        }
    }
    tmp = arr[i + 1];
    arr[i + 1] = arr[high];
    arr[high] = tmp;
    return (i + 1);
}

inline void quickSort(meeting_unit *arr, int low, int high){
    if (low < high){
        /* pi is partitioning index, arr[p] is now
           at right place */
        int pi = partition(arr, low, high);
        // Separately sort elements before
        // partition and after partition
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

inline bool isvalid(int p,int r, int size){
	return 0 <= p && p< r &&r < size;
}

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
		for(size_t bid=start;bid<end;bid++){

			if(bench->config->use_hash){
				size_t kHashTableCapacity = bench->config->num_meeting_buckets*bench->config->bucket_size;
				meeting_unit *meet = bench->meeting_buckets[!bench->current_bucket]+bid;
				meeting_unit *new_tab = bench->meeting_buckets[bench->current_bucket];
				if(meet->key==ULL_MAX){
					continue;
				}
				size_t slot = meet->key%kHashTableCapacity;
				bool exist = false;
				int ite = 0;
				while (true){
					if (new_tab[slot].key == meet->key){
						new_tab[slot].start = meet->start;
						exist = true;
						break;
					}
					if (new_tab[slot].key == ULL_MAX){
						break;
					}
					slot = (slot + 1)%kHashTableCapacity;
				}
				if(!exist&&bench->cur_time - meet->start>=bench->config->min_meet_time){
					meetings[meeting_index++] = *meet;
					if(meeting_index==200){
						bench->batch_meet(meetings, 200);
						meeting_index = 0;
					}
				}
				continue;
			}
			meeting_unit *bucket_new = bench->meeting_buckets[bench->current_bucket]+bid*bench->config->bucket_size;
			meeting_unit *bucket_old = bench->meeting_buckets[!bench->current_bucket]+bid*bench->config->bucket_size;

			uint size_new = min(bench->meeting_buckets_counter[bench->current_bucket][bid],bench->config->bucket_size);
			uint size_old = min(bench->meeting_buckets_counter[!bench->current_bucket][bid],bench->config->bucket_size);


			if(bench->config->brute_meeting){
				for(uint i=0;i<size_old;i++){
					bool updated = false;
					for(uint j=0;j<size_new;j++){
						if(bucket_old[i].pid1==bucket_new[j].pid1){
							bucket_new[j].start = bucket_old[i].start;
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
			}else{
				if(size_new>1){
					quickSort(bucket_new,0,size_new-1);
				}
				uint i=0,j=0;
				for(;i<size_old;i++){
					bool updated = false;
					for(;j<size_new;j++){
						if(bucket_old[i].pid1==bucket_new[j].pid1){
							bucket_new[j].start = bucket_old[i].start;
							updated = true;
							break;
						}else if(bucket_old[i].pid1<bucket_new[j].pid1){
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

//				if(size_new==13){
//					printf("%ld %d %d %d\n",bid,size_old,size_new,live);
//					for(int k=0;k<size_old;k++){
//						printf("%d ",bucket_old[k].pid1);
//					}
//					printf("\n");
//					for(int k=0;k<size_new;k++){
//						printf("%d ",bucket_new[k].pid1);
//					}
//					printf("\n");
//				}

				// reset the old buckets for next batch of processing
				bench->meeting_buckets_counter[!bench->current_bucket][bid] = 0;
			}

		}
	}
	bench->batch_meet(meetings, meeting_index);
	delete []meetings;
	return NULL;
}

void workbench::update_meetings(){
	struct timeval start = get_cur_time();

	query_context tctx;
	tctx.config = config;
	tctx.target[0] = (void *)this;
	tctx.num_units = config->num_meeting_buckets;
	if(config->use_hash){
		tctx.num_units = config->num_meeting_buckets*config->bucket_size;
	}
	pthread_t threads[tctx.config->num_threads];

	meeting_counter = 0;
	for(int i=0;i<tctx.config->num_threads;i++){
		pthread_create(&threads[i], NULL, update_meetings_unit, (void *)&tctx);
	}
	for(int i = 0; i < tctx.config->num_threads; i++ ){
		void *status;
		pthread_join(threads[i], &status);
	}
	uint mc = 0;
	if(config->use_hash){
		for(size_t i=0;i<(size_t)config->num_meeting_buckets*config->bucket_size;i++){
			mc += (meeting_buckets[current_bucket][i].key!=ULL_MAX);
		}
	}else{
		for(int i=0;i<config->num_meeting_buckets;i++){
			mc += meeting_buckets_counter[current_bucket][i];
		}
	}
	logt("update meeting: %d meetings active, %d meeting identified",start,mc,meeting_counter);
}

