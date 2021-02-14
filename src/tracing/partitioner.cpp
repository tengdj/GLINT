/*
 * partitioner.cpp
 *
 *  Created on: Feb 1, 2021
 *      Author: teng
 */

#include "trace.h"


partition_info::partition_info(size_t ng, size_t no, size_t zs){
	num_grids = ng;
	num_objects = no;
	zone_size = zs;

	// make the capacity half more than the grid number
	capacity = num_grids+num_grids/2;
	buffer_zones = new uint[zone_size*capacity];
	cur_zone = new uint[num_grids];
	grid_checkings = new uint[2*5*num_objects];
	pthread_mutex_init(&lk,NULL);
	for(int i=0;i<50;i++){
		pthread_mutex_init(&insert_lk[i],NULL);
	}
}

bool partition_info::resize(size_t newcapacity){
	if(newcapacity<capacity){
		return false;
	}
	uint *npart = new uint[zone_size*newcapacity];
	memcpy(npart,buffer_zones,zone_size*capacity*sizeof(uint));
	delete []buffer_zones;
	buffer_zones = npart;
	capacity = newcapacity;
	return true;
}

bool partition_info::insert(uint gid, uint pid){
	pthread_mutex_lock(&insert_lk[gid%50]);
	uint cur_loc = get_zone_size(cur_zone[gid]);
	// one buffer zone is full or no zone is assigned, locate next buffer zone
	if(cur_loc==0){
		pthread_mutex_lock(&lk);
		// enlarge the buffer zone by 50% if the capacity is reached
		if(cur_free_zone==capacity){
			enlarge();
		}
		// link to the previous zone offset
		buffer_zones[(zone_size+1)*cur_free_zone] = cur_zone[gid];
		cur_zone[gid] = cur_free_zone++;
		pthread_mutex_unlock(&lk);
	}
	// the first number in the buffer zone links to the previous zone (0 if no one)
	// the second number in the buffer zone is the number of objects in current zone
	buffer_zones[(zone_size+1)*cur_zone[gid]+2+cur_loc] = pid;
	buffer_zones[(zone_size+1)*cur_zone[gid]+1]++;
	pthread_mutex_unlock(&insert_lk[gid%50]);
	return true;
}

bool partition_info::check(uint gid, uint pid){
	uint cur_offset = cur_zone[gid];
	do{
		assert(num_grid_checkings<2*5*num_objects);
		grid_checkings[2*num_grid_checkings] = pid;
		grid_checkings[2*num_grid_checkings+1] = cur_offset;
		num_grid_checkings++;
		// the first number in this region point to the previous zone
		cur_offset = buffer_zones[(zone_size+1)*cur_offset];
	}while(cur_offset!=0);
	return true;
}


partition_info *grid_partitioner::partition(Point *points, size_t num_objects){
	struct timeval start = get_cur_time();
	clear();

	size_t num_grids = grid->get_grid_num();
	if(!pinfo||pinfo->num_grids!=num_grids||pinfo->num_objects!=num_objects){
		if(pinfo){
			delete pinfo;
		}
		pinfo = new partition_info(num_grids, num_objects, config.max_objects_per_grid);
	}
	pinfo->points = points;

	double x_buffer = config.reach_distance*degree_per_meter_longitude(grid->space.low[1]);
	double y_buffer = config.reach_distance*degree_per_meter_latitude;

	// assign each object to proper grids
	for(int pid=0;pid<num_objects;pid++){
		Point *p = points+pid;
		size_t gid = grid->getgridid(p);
		pinfo->insert(gid,pid);
	}
	logt("partition %ld objects into %ld grids",start,num_objects, num_grids);

	// the query process
	// each point is associated with a list of grids
	for(size_t pid=0;pid<num_objects;pid++){
		size_t gid_code = grid->getgrids(points+pid,x_buffer,y_buffer);
		size_t gid = gid_code>>4;
		bool top_right = gid_code & 8;
		bool right = gid_code & 4;
		bool bottom_right = gid_code &2;
		bool top = gid_code &1;
		pinfo->check(gid,pid);
		if(top_right){
			pinfo->check(gid+grid->dimx+1, pid);
		}
		if(right){
			pinfo->check(gid+1,pid);
		}
		if(top){
			pinfo->check(gid+grid->dimx,pid);
		}
		if(bottom_right){
			pinfo->check(gid-grid->dimx+1,pid);
		}
	}

	logt("pack",start);
	return pinfo;
}


partition_info *qtree_partitioner::partition(Point *points, size_t num_objects){
	struct timeval start = get_cur_time();
	clear();
	qtree = new QTNode(mbr, &qconfig);
	qconfig.points = points;

	for(uint pid=0;pid<num_objects;pid++){
		//log("%d",pid);
		qtree->insert(pid);
	}
	// set the ids and other stuff
	qtree->finalize();
	size_t num_grids = qtree->leaf_count();
	logt("partition into %d grids",start,num_grids);

	if(!pinfo||pinfo->num_grids!=num_grids||pinfo->num_objects!=num_objects){
		if(pinfo){
			delete pinfo;
		}
		pinfo = new partition_info(num_grids, num_objects, config.max_objects_per_grid);
	}
	pinfo->points = points;

	//qtree->pack_data(partitions, os);
	//qtree->print();
	logt("pack",start);

	// tree lookups
	vector<uint> nodes;
	for(size_t pid=0;pid<num_objects;pid++){
		qtree->query(nodes,points+pid);
		for(uint gid:nodes){
			pinfo->check(gid,pid);
		}
		nodes.clear();
	}

	logt("lookup",start);
	return pinfo;
}


void qtree_partitioner::clear(){
	if(qtree){
		delete qtree;
		qtree = NULL;
	}
}

