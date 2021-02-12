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

class partitioner{
protected:
	configuration config;
	box mbr;
public:
	partitioner(){}
	virtual ~partitioner(){};

	virtual void clear() = 0;
	virtual query_context partition(Point *objects, size_t num_objects) = 0;
	//void pack_grids(query_context &);
};

class grid_partitioner:public partitioner{
	Grid *grid = NULL;
public:
	grid_partitioner(box &m, configuration &conf){
		mbr = m;
		config = conf;
		grid = new Grid(mbr, config.grid_width);
	}
	~grid_partitioner(){
		if(grid){
			delete grid;
		}
	};
	void clear(){};
	query_context partition(Point *objects, size_t num_objects);
};

class qtree_partitioner:public partitioner{
	QTNode *qtree = NULL;
	QConfig qconfig;
public:
	qtree_partitioner(box &m, configuration &conf){
		mbr = m;
		config = conf;
		qconfig.grid_width = config.grid_width;
		qconfig.max_objects = config.max_objects_per_grid;
		qconfig.x_buffer = config.reach_distance*degree_per_meter_longitude(mbr.low[1]);
		qconfig.y_buffer = config.reach_distance*degree_per_meter_latitude;
	}
	~qtree_partitioner(){
		clear();
		if(qtree){
			delete qtree;
		}
	}
	void clear();
	query_context partition(Point *objects, size_t num_objects);
};



#endif /* SRC_TRACING_PARTITIONER_H_ */
