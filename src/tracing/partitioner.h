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


// the data structure used to pack
// the partition information
class partition_info{
	// next available free partition buffer zone
	size_t cur_free_zone = 1;

	// total number of partition buffer zone
	size_t capacity = 0;
	// size of each partition buffer zone
	size_t zone_size = 0;
	pthread_mutex_t lk;
	pthread_mutex_t insert_lk[50];

	// the pool of maintaining objects assignment
	// each buffer zone: |num_addition_zones|add_zone_id1...add_zone_idn|num_points|point_id1...point_idn|
	uint *buffer_zones = NULL;

	// which zone each grid is packing into
	uint *cur_zone = NULL;
public:

	uint *grid_checkings = NULL;
	size_t num_grid_checkings = 0;
	size_t num_grids = 0;

	// external source
	Point *points = NULL;
	size_t num_objects = 0;


	partition_info(size_t ng, size_t no, size_t zs);
	~partition_info(){
		delete []buffer_zones;
		delete []cur_zone;
		delete []grid_checkings;
	}
	bool resize(size_t new_capacity);
	bool enlarge(){
		return resize(1.5*capacity);
	}
	// insert point pid to grid gid
	bool insert(uint gid, uint pid);
	bool batch_insert(uint gid, uint num_objects, uint *pids);
	bool check(uint gid, uint pid);

	// get the object-grid checking pairs
	uint *get_grid_check();
	void clear(){
		memset(cur_zone,0,sizeof(uint)*num_grids);
		// just reset the cursor, no need to reset the content in the buffer_zone
		cur_free_zone = 1;
		num_grid_checkings = 0;
	}
	inline uint *get_zone(uint zid){
		return buffer_zones + zid*(zone_size+2)+2;
	}
	inline uint get_zone_size(uint zid){
		return *(buffer_zones + zid*(zone_size+2)+1);
	}
	inline uint get_prev_zoneid(uint zid){
		return *(buffer_zones + zid*(zone_size+2));
	}
	inline void set_zone_size(uint zid, uint s){
		*(buffer_zones + zid*(zone_size+2)+1) = s;
	}
	inline void increase_zone_size(uint zid){
		*(buffer_zones + zid*(zone_size+2)+1) += 1;
	}
	inline void set_prev_zoneid(uint zid, uint prev_zid){
		*(buffer_zones + zid*(zone_size+2)) = prev_zid;
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
	//void pack_grids(query_context &);
};

class grid_partitioner:public partitioner{
	Grid *grid = NULL;
public:
	grid_partitioner(box &m, configuration &conf){
		mbr = m;
		config = conf;
		grid = new Grid(mbr, config.grid_width);
		//grid->print();
		pinfo = new partition_info(grid->get_grid_num(),config.num_objects, max(config.grid_capacity,config.num_objects/grid->get_grid_num()));
	}
	~grid_partitioner(){
		delete grid;
		delete pinfo;
	};
	void clear(){};
	partition_info *partition(Point *objects, size_t num_objects);
};

class qtree_partitioner:public partitioner{
	QTNode *qtree = NULL;
	QConfig qconfig;
public:
	qtree_partitioner(box &m, configuration &conf){
		mbr = m;
		config = conf;
		qconfig.grid_width = config.grid_width;
		qconfig.max_objects = config.grid_capacity;
		qconfig.x_buffer = config.reach_distance*degree_per_meter_longitude(mbr.low[1]);
		qconfig.y_buffer = config.reach_distance*degree_per_meter_latitude;
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
};



#endif /* SRC_TRACING_PARTITIONER_H_ */
