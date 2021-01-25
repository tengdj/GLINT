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
	double *data = NULL;
	uint *offset_size = NULL;
	int *result = NULL;
	~query_context(){
		if(data){
			delete data;
		}
		if(offset_size){
			delete offset_size;
		}
		if(result){
			delete result;
		}
	}
};



#endif /* QUERY_CONTEXT_H_ */
