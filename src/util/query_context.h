/*
 * query_context.h
 *
 *  Created on: Jan 25, 2021
 *      Author: teng
 */

#ifndef QUERY_CONTEXT_H_
#define QUERY_CONTEXT_H_
#include "config.h"
#include <pthread.h>

#define MAX_LOCK_NUM 100

class offset_size{
public:
	uint offset;
	uint size;
};

class query_context{
	size_t next_report = 0;
	size_t step = 0;
	size_t counter = 0;

public:
	configuration config;
	size_t num_objects = 0;
	size_t batch_size = 10;
	size_t report_gap = 10;
	pthread_mutex_t lk[MAX_LOCK_NUM];

	// query source
	void *target[4] = {NULL,NULL,NULL,NULL};

	// query results
	size_t checked = 0;
	size_t found = 0;

	~query_context(){
	}
	query_context(){
		for(int i=0;i<MAX_LOCK_NUM;i++){
			pthread_mutex_init(&lk[i], NULL);
		}
	}
	void lock(int hashid=0){
		pthread_mutex_lock(&lk[hashid%MAX_LOCK_NUM]);
	}
	void unlock(int hashid=0){
		pthread_mutex_unlock(&lk[hashid%MAX_LOCK_NUM]);
	}
	bool next_batch(size_t &start, size_t &end){
		int gt = 0;
		pthread_mutex_lock(&lk[0]);
		start = counter;
		counter += batch_size;
		end = counter;
		if(end>num_objects){
			end = num_objects;
		}
		//log("%d %d %d %d",start,next_report,num_objects,report_gap);
		if(start>next_report&&start<=num_objects){
			log("%ld%%",start*100/num_objects);
			next_report += num_objects*report_gap/100;
		}

		pthread_mutex_unlock(&lk[0]);
		return start<num_objects;
	}
	void idle(){
		pthread_mutex_lock(&lk[0]);
		counter--;
		pthread_mutex_unlock(&lk[0]);
	}
	void busy(){
		pthread_mutex_lock(&lk[0]);
		counter++;
		pthread_mutex_unlock(&lk[0]);
	}
	bool all_idle(){
		return counter == 0;
	}
	void clear(){
//		for(int i=0;i<4;i++){
//			target[i] = NULL;
//		}
		next_report = 0;
		counter = 0;
	}
};



#endif /* QUERY_CONTEXT_H_ */
