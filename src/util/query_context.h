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
public:
	configuration config;
	void *target[4] = {NULL,NULL,NULL,NULL};
	double *data = NULL;
	uint *offset_size = NULL;
	int *result = NULL;
	size_t counter = 0;
	pthread_mutex_t lk[MAX_LOCK_NUM];
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
		gt = --counter;
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
};



#endif /* QUERY_CONTEXT_H_ */
