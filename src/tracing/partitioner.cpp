/*
 * partitioner.cpp
 *
 *  Created on: Feb 1, 2021
 *      Author: teng
 */

#include "trace.h"

query_context grid_partitioner::partition(Point *points, uint *pids, size_t num_objects){
	struct timeval start = get_cur_time();
	clear();

	vector<vector<uint>> grids;
	grids.resize(grid->get_grid_num()+1);

	vector<vector<uint>> exact_grids;
	exact_grids.resize(grid->get_grid_num()+1);

	double x_buffer = config.reach_distance*degree_per_meter_longitude(grid->space.low[1]);
	double y_buffer = config.reach_distance*degree_per_meter_latitude;

	// pick one object for generating
	for(int pid=0;pid<num_objects;pid++){
		Point *p = points+pid;
		size_t gid_code = grid->getgrids(p,x_buffer,y_buffer);
		size_t gid = gid_code>>4;
		bool top_right = gid_code & 8;
		bool right = gid_code & 4;
		bool bottom_right = gid_code &2;
		bool top = gid_code &1;
		grids[gid].push_back(pid);
		exact_grids[gid].push_back(pid);
		if(top_right){
			grids[gid+grid->dimx+1].push_back(pid);
		}
		if(right){
			grids[gid+1].push_back(pid);
		}
		if(top){
			grids[gid+grid->dimx].push_back(pid);
		}
		if(bottom_right){
			grids[gid-grid->dimx+1].push_back(pid);
		}
	}
	size_t total_objects = 0;
	size_t num_grids = 0;
	for(vector<uint> &gs:grids){
		if(gs.size()>0){
			total_objects += gs.size();
			num_grids++;
		}
	}
	logt("space is partitioned into %ld grids %ld objects",start,num_grids,total_objects);


	uint *gridassign = new uint[num_objects];
	uint *data = new uint[total_objects];
	offset_size *os = new offset_size[num_grids];
	uint *result = new uint[num_grids];

	// pack the data
	uint curnode = 0;
	size_t calculation = 0;
	for(int i=0;i<grids.size();i++){
		if(grids[i].size()>0){
			if(curnode==0){
				os[curnode].offset = 0;
			}else{
				os[curnode].offset = os[curnode-1].offset+os[curnode-1].size;
			}
			os[curnode].size = grids[i].size();
			uint *cur_data = data + os[curnode].offset;
			for(int j=0;j<grids[i].size();j++){
				cur_data[j] = grids[i][j];
			}

			for(int j=0;j<exact_grids[i].size();j++){
				gridassign[exact_grids[i][j]] = curnode;
				calculation += grids[i].size();
			}
			curnode++;
		}
	}

	//pack the grids into array
	query_context ctx;
	ctx.config = config;
	ctx.target[0] = (void *)points;
	ctx.target_length[0] = num_objects;
	ctx.target[1] = (void *)data;
	ctx.target_length[1] = total_objects;
	ctx.target[2] = (void *)os;
	ctx.target_length[2] = num_grids;
	ctx.target[3] = (void *)result;
	ctx.target_length[3] = num_grids;
	ctx.num_objects = num_grids;
	ctx.target[4] = (void *)gridassign;
	ctx.target_length[4] = num_objects;

	logt("packed into %ld grids %ld objects %ld calculation",start,num_grids,total_objects,calculation);
	return ctx;
}


query_context qtree_partitioner::partition(Point *points, uint *pids, size_t num_objects){
	struct timeval start = get_cur_time();
	clear();
	qtree = new QTNode(mbr, &qconfig);
	qconfig.points = points;

	for(uint pid=0;pid<num_objects;pid++){
		//log("%d",pid);
		qtree->insert(pid);
	}
	//qtree->get_leafs(grids, true);
	size_t total_objects = qtree->num_objects();
	size_t num_grids = qtree->leaf_count();
	logt("space is partitioned into %d grids %ld objects",start,num_grids,total_objects);

	uint *grid_assignment = new uint[num_objects];
	uint *data = new uint[total_objects];
	offset_size *os = new offset_size[num_grids];
	uint *result = new uint[total_objects];
	uint curnode = 0;
	qtree->pack_data(grid_assignment, data, os, curnode);
	size_t calculation = 0;
	for(int i=0;i<num_objects;i++){
		calculation += os[grid_assignment[i]].size;
	}
	//qtree->print();

	//pack the grids into array
	query_context ctx;
	ctx.config = config;
	ctx.target[0] = (void *)points;
	ctx.target_length[0] = num_objects;
	ctx.target[1] = (void *)data;
	ctx.target_length[1] = total_objects;
	ctx.target[2] = (void *)os;
	ctx.target_length[2] = num_grids;
	ctx.target[3] = (void *)result;
	ctx.target_length[3] = num_objects;
	ctx.target[4] = (void *)grid_assignment;
	ctx.target_length[4] = num_objects;

	ctx.num_objects = num_objects;
	logt("packed into %ld grids %ld objects %ld calculation",start,num_grids,total_objects,calculation);

	//qtree->print();
	return ctx;
}


void qtree_partitioner::clear(){
	if(qtree){
		delete qtree;
		qtree = NULL;
	}
}

