/*
 * partitioner.cpp
 *
 *  Created on: Feb 1, 2021
 *      Author: teng
 */

#include "trace.h"

query_context grid_partitioner::partition(Point *points, size_t num_objects){
	struct timeval start = get_cur_time();
	clear();

	size_t num_grids = grid->get_grid_num();

	vector<vector<uint>> grids;
	grids.resize(num_grids);

	double x_buffer = config.reach_distance*degree_per_meter_longitude(grid->space.low[1]);
	double y_buffer = config.reach_distance*degree_per_meter_latitude;

	// pick one object for generating
	for(int pid=0;pid<num_objects;pid++){
		Point *p = points+pid;
		size_t gid = grid->getgridid(p);
		grids[gid].push_back(pid);
	}
	logt("partition %ld objects into %ld grids",start,num_objects, num_grids);

	uint *partitions = new uint[num_objects];
	offset_size *os = new offset_size[num_grids];
	uint *result = new uint[num_objects];
	uint *grid_checks = new uint[2*5*num_objects];
	size_t num_checks = 0;

	//pack the grids into array
	for(size_t i=0;i<num_grids;i++){
		if(i==0){
			os[i].offset = 0;
		}else{
			os[i].offset = os[i-1].offset+os[i-1].size;
		}
		os[i].size = grids[i].size();
		uint *cur_part = partitions + os[i].offset;
		for(int j=0;j<grids[i].size();j++){
			cur_part[j] = grids[i][j];
		}
	}

	// the query process
	// each point is associated with a list of grids
	for(size_t pid=0;pid<num_objects;pid++){
		size_t gid_code = grid->getgrids(points+pid,x_buffer,y_buffer);
		size_t gid = gid_code>>4;
		bool top_right = gid_code & 8;
		bool right = gid_code & 4;
		bool bottom_right = gid_code &2;
		bool top = gid_code &1;
		grid_checks[2*num_checks] = pid;
		grid_checks[2*num_checks+1] = gid;
		num_checks++;
		if(top_right){
			grid_checks[2*num_checks] = pid;
			grid_checks[2*num_checks+1] = gid+grid->dimx+1;
			num_checks++;
		}
		if(right){
			grid_checks[2*num_checks] = pid;
			grid_checks[2*num_checks+1] = gid+1;
			num_checks++;
		}
		if(top){
			grid_checks[2*num_checks] = pid;
			grid_checks[2*num_checks+1] = gid+grid->dimx;
			num_checks++;
		}
		if(bottom_right){
			grid_checks[2*num_checks] = pid;
			grid_checks[2*num_checks+1] = gid-grid->dimx+1;
			num_checks++;
		}
	}

	// form a query context object
	query_context ctx;
	ctx.config = config;
	ctx.target[0] = (void *)points;
	ctx.target_length[0] = num_objects;
	ctx.target[1] = (void *)partitions;
	ctx.target_length[1] = num_objects;
	ctx.target[2] = (void *)os;
	ctx.target_length[2] = num_grids;
	ctx.target[3] = (void *)result;
	ctx.target_length[3] = num_objects;
	ctx.target[4] = (void *)grid_checks;
	ctx.target_length[4] = num_checks;

	ctx.num_objects = num_checks;
	logt("pack",start);
	return ctx;
}


query_context qtree_partitioner::partition(Point *points, size_t num_objects){
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
	logt("partition %ld objects into %d grids",start,num_objects,num_grids);

	uint *partitions = new uint[num_objects];
	offset_size *os = new offset_size[num_grids];
	uint *result = new uint[num_objects];
	uint *grid_checks = new uint[2*5*num_objects];
	size_t num_checks = 0;
	qtree->pack_data(partitions, os);
	//qtree->print();

	// tree lookups
	vector<uint> nodes;
	for(size_t pid=0;pid<num_objects;pid++){
		qtree->query(nodes,points+pid);
		for(uint gid:nodes){
			grid_checks[2*num_checks] = pid;
			grid_checks[2*num_checks+1] = gid;
			num_checks++;
		}
		nodes.clear();
	}

	//pack the grids into array
	query_context ctx;
	ctx.config = config;
	ctx.target[0] = (void *)points;
	ctx.target_length[0] = num_objects;
	ctx.target[1] = (void *)partitions;
	ctx.target_length[1] = num_objects;
	ctx.target[2] = (void *)os;
	ctx.target_length[2] = num_grids;
	ctx.target[3] = (void *)result;
	ctx.target_length[3] = num_objects;
	ctx.target[4] = (void *)grid_checks;
	ctx.target_length[4] = num_checks;

	ctx.num_objects = num_checks;
	logt("pack",start);

	//qtree->print();
	return ctx;
}


void qtree_partitioner::clear(){
	if(qtree){
		delete qtree;
		qtree = NULL;
	}
}

