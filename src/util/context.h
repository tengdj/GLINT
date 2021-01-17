/*
 * context.h
 *
 *  Created on: Jan 16, 2021
 *      Author: teng
 */

#ifndef SRC_UTIL_CONTEXT_H_
#define SRC_UTIL_CONTEXT_H_

#include "util.h"

class context{
public:
	int thread_id = 0;
	int thread_num = get_num_threads();
	void *target = NULL;
};

#endif /* SRC_UTIL_CONTEXT_H_ */
