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
	vector<vector<Point *>> grids;
	configuration config;
	box mbr;
public:
	partitioner(){}
	virtual ~partitioner(){};

	void clear(){
		for(vector<Point *> &ps:grids){
			ps.clear();
		}
	};
	virtual void index(Point *objects, size_t num_objects) = 0;
	virtual void partition(Point *objects, size_t num_objects) = 0;
	vector<vector<Point *>> get_grids(){
		return grids;
	}
};

class grid_partitioner:public partitioner{
	Grid *grid = NULL;
public:
	grid_partitioner(box &m, configuration &conf){
		mbr = m;
		config = conf;
	}
	~grid_partitioner(){
		clear();
		if(grid){
			delete grid;
		}
	};
	void index(Point *objects, size_t num_objects);
	void partition(Point *objects, size_t num_objects);
};

class qtree_partitioner:public partitioner{
	QTNode *qtree = NULL;
	QConfig qconfig;
public:
	qtree_partitioner(box &m, configuration &conf){
		mbr = m;
		config = conf;
	}
	~qtree_partitioner(){
		clear();
		if(qtree){
			delete qtree;
		}
	}
	void clear();
	void index(Point *objects, size_t num_objects);
	void partition(Point *objects, size_t num_objects);
};



#endif /* SRC_TRACING_PARTITIONER_H_ */
