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
protected:
	configuration config;
	box mbr;
public:
	partitioner(){}
	virtual ~partitioner(){};

	virtual void clear() = 0;
	virtual void partition(workbench *bench) = 0;
	virtual void lookup(workbench *bench, uint start_pid) = 0;
	virtual workbench *build_schema(Point *objects, size_t num_objects) = 0;
};

class qtree_partitioner:public partitioner{
public:
	qtree_partitioner(box &m, configuration &conf){
		mbr = m;
		config = conf;
	}
	~qtree_partitioner(){
	}
	void clear(){};
	void partition(workbench *bench);
	void lookup(workbench *bench, uint start_pid);
	workbench * build_schema(Point *objects, size_t num_objects);
};



#endif /* SRC_TRACING_PARTITIONER_H_ */
