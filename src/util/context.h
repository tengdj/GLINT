/*
 * context.h
 *
 *  Created on: Jan 16, 2021
 *      Author: teng
 */

#ifndef SRC_UTIL_CONTEXT_H_
#define SRC_UTIL_CONTEXT_H_

#include "util.h"

enum PROCESS_METHOD{
	QTREE = 0,
	GPU = 1
};

class context{
public:
	int thread_id = 0;
	PROCESS_METHOD method = QTREE;
	int duration = 0;
	int num_threads = get_num_threads();
	int num_objects = 0;
	int num_grids = 0;
	int num_trips = 0;
	void *target[3] = {NULL,NULL,NULL};
	context(){
	}

};

#endif /* SRC_UTIL_CONTEXT_H_ */
