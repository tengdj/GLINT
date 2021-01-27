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
	~query_context(){
	}
	query_context(){}
};



#endif /* QUERY_CONTEXT_H_ */
