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
class query_context{
	size_t max_counter = 0;
	size_t next_report = 0;
	size_t step = 0;
public:
	configuration config;
	size_t counter = 0;
	int report_gap = 1;
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
	int fetch_one(){
		int gt = 0;
		pthread_mutex_lock(&lk[0]);
		if(step == 0){
			step = counter/report_gap;
			next_report = counter - step;
			max_counter = counter;
		}
		gt = --counter;
		if(counter==next_report){
			log("%d%%",(max_counter-next_report)*100/max_counter);
			next_report -= step;
		}
		pthread_mutex_unlock(&lk[0]);
		return gt;
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
	void clear(){
		for(int i=0;i<4;i++){
			if(target[i]!=NULL){
				free(target[i]);
				target[i] = NULL;
			}
		}
	}
};



#endif /* QUERY_CONTEXT_H_ */
