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
	pthread_mutex_t insert_lk[50];
public:
	configuration config;

	// the pool of maintaining objects assignment
	// each grid buffer: |num_objects|point_id1...point_idn|
	uint *grids = NULL;
	size_t num_grids = 0;

	// the QTree schema
	QTSchema *schema = NULL;
	uint num_nodes = 0;

	// the space for storing the point-grid pairs
	checking_unit *checking_units = NULL;
	uint num_checking_units = 0;
	size_t checking_units_capacity = 0;

	// the processing stack for looking up
	uint *lookup_stack[2] = {NULL, NULL};
	uint stack_index[2] = {0,0};

	// external source
	Point *points = NULL;

	partition_info(configuration conf);
	// for container
	~partition_info(){
		if(grids){
			delete []grids;
		}
		if(checking_units){
			delete []checking_units;
		}
		if(schema){
			delete []schema;
		}
	}

	// insert point pid to grid gid
	bool insert(uint gid, uint pid);
	bool batch_insert(uint gid, uint num_objects, uint *pids);
	bool check(uint gid, uint pid);
	bool batch_check(checking_unit *cu, uint num_cu);


	void claim_space(uint ng){
		assert(ng>0);
		if(num_grids != ng){
			if(grids){
				delete []grids;
			}
			num_grids = ng;
			grids = new uint[(config.grid_capacity+1)*num_grids];
		}

		if(!checking_units){
			checking_units = new checking_unit[checking_units_capacity];
		}
		reset();
	}
	void reset(){
		// reset the number of objects in each grid
		for(int i=0;i<num_grids;i++){
			grids[i*(config.grid_capacity+1)] = 0;
		}
		num_checking_units = 0;
	}
	inline uint get_grid_size(uint gid){
		return grids[gid*(config.grid_capacity+1)];
	}
	inline uint *get_grid(uint gid){
		return grids + gid*(config.grid_capacity+1)+1;
	}
	inline void reset_grid(uint gid){
		grids[gid*(config.grid_capacity+1)] = 0;
	}
	inline void increase_grid_size(uint gid){
		grids[gid*(config.grid_capacity+1)]++;
	}
};

class partitioner{
protected:
	configuration config;
	box mbr;
public:
	partition_info *pinfo = NULL;
	partitioner(){}
	virtual ~partitioner(){};

	virtual void clear() = 0;
	virtual void partition(Point *objects, size_t num_objects) = 0;
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
	void partition(Point *objects, size_t num_objects);
	void build_schema(Point *objects, size_t num_objects);
};


class qtree_partitioner:public partitioner{
public:
	qtree_partitioner(box &m, configuration &conf){
		mbr = m;
		config = conf;
	}
	~qtree_partitioner(){
		if(pinfo){
			delete pinfo;
		}
	}
	void clear(){};
	void partition(Point *objects, size_t num_objects);
	void build_schema(Point *objects, size_t num_objects);
};



#endif /* SRC_TRACING_PARTITIONER_H_ */
