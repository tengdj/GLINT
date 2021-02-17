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


typedef struct checking_unit{
	uint pid;
	uint gid;
	uint offset;
}checking_unit;

// the data structure used to pack
// the partition information
class partition_info{
public:
	pthread_mutex_t insert_lk[50];

	// the pool of maintaining objects assignment
	// each grid buffer: |num_objects|point_id1...point_idn|
	uint *grids = NULL;
	size_t grid_capacity = 0;
	size_t num_grids = 0;
	// the size of a processing unit
	size_t unit_size = 0;

	// the QTree schema
	QTSchema *schema = NULL;
	uint num_nodes = 0;

	checking_unit *checking_units = NULL;
	size_t num_checking_units = 0;
	size_t checking_units_capacity = 0;

	// external source
	Point *points = NULL;
	size_t num_objects = 0;


	partition_info(size_t ng, size_t no, size_t gs, size_t us);
	~partition_info(){
		delete []grids;
		delete []checking_units;
	}

	// insert point pid to grid gid
	bool insert(uint gid, uint pid);
	bool batch_insert(uint gid, uint num_objects, uint *pids);
	bool check(uint gid, uint pid);
	bool batch_check(checking_unit *cu, uint num_cu);

	void clear(){
		// reset the number of objects in each grid
		for(int i=0;i<num_grids;i++){
			grids[i*(grid_capacity+1)] = 0;
		}
		num_checking_units = 0;
	}
	inline uint get_grid_size(uint gid){
		return grids[gid*(grid_capacity+1)];
	}
	inline uint *get_grid(uint gid){
		return grids + gid*(grid_capacity+1)+1;
	}
	inline void reset_grid(uint gid){
		grids[gid*(grid_capacity+1)] = 0;
	}
	inline void increase_grid_size(uint gid){
		grids[gid*(grid_capacity+1)]++;
	}
};

class partitioner{
protected:
	configuration config;
	box mbr;
	partition_info *pinfo = NULL;
public:
	partitioner(){}
	virtual ~partitioner(){};

	virtual void clear() = 0;
	virtual partition_info * partition(Point *objects, size_t num_objects) = 0;
	virtual void build_schema(Point *objects, size_t num_objects) = 0;
	//void pack_grids(query_context &);
};

class grid_partitioner:public partitioner{
	Grid *grid = NULL;
public:
	grid_partitioner(box &m, configuration &conf){
		mbr = m;
		config = conf;
	}
	~grid_partitioner(){
		if(grid){
			delete grid;
		}
		if(pinfo){
			delete pinfo;
		}
	};
	void clear(){};
	partition_info *partition(Point *objects, size_t num_objects);
	void build_schema(Point *objects, size_t num_objects){
		if(grid){
			delete grid;
		}
		grid = new Grid(mbr, config.grid_width);
	}
};


class qtree_partitioner:public partitioner{
	QTNode *qtree = NULL;
	QTSchema *schema = NULL;
	uint num_nodes = 0;
	uint num_grids = 0;
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
		if(pinfo){
			delete pinfo;
		}
	}
	void clear();
	partition_info *partition(Point *objects, size_t num_objects);
	void build_schema(Point *objects, size_t num_objects);
};



#endif /* SRC_TRACING_PARTITIONER_H_ */
