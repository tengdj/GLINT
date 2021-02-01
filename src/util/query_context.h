/*
 * query_context.h
 *
 *  Created on: Jan 25, 2021
 *      Author: teng
 */

#ifndef QUERY_CONTEXT_H_
#define QUERY_CONTEXT_H_
#include "config.h"

class query_context{
public:
	configuration config;
	void *target[3] = {NULL,NULL,NULL};
	double *data = NULL;
	uint *offset_size = NULL;
	int *result = NULL;
	size_t counter = 0;
	pthread_mutex_t lk;
	~query_context(){
	}
	query_context(){
		pthread_mutex_init(&lk, NULL);
	}
	int fetch_one(){
		int gt = 0;
		pthread_mutex_lock(&lk);
		gt = --counter;
		pthread_mutex_unlock(&lk);
		return gt;
	}

};



#endif /* QUERY_CONTEXT_H_ */
