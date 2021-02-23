/*
 * partitioner.h
 *
 *  Created on: Feb 1, 2021
 *      Author: teng
 */

#ifndef SRC_TRACING_PARTITIONER_H_
#define SRC_TRACING_PARTITIONER_H_

#include "../util/config.h"
#include "../geometry/geometry.h"
#include "../index/QTree.h"
#include "workbench.h"


class partitioner{
	configuration *config = NULL;
	box mbr;
public:
	partitioner(box &m, configuration *conf){
		mbr = m;
		config = conf;
	}
	~partitioner(){
	}
	void clear(){};
	workbench * build_schema(Point *objects, size_t num_objects);
};



#endif /* SRC_TRACING_PARTITIONER_H_ */
