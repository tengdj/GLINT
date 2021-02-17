/*
 * partitioner.cpp
 *
 *  Created on: Feb 1, 2021
 *      Author: teng
 */

#include "trace.h"


partition_info::partition_info(size_t ng, size_t no, size_t gc, size_t us){
	num_grids = ng;
	num_objects = no;
	grid_capacity = gc;
	unit_size = us;
	// each object can be checked with up to 5 grids, each with several checking units
	checking_units_capacity = 5*num_objects*(grid_capacity/unit_size+1);
	// make the capacity of zones half more than the grid number
	grids = new uint[(grid_capacity+1)*num_grids];
	checking_units = new checking_unit[checking_units_capacity];
	clear();
	for(int i=0;i<50;i++){
		pthread_mutex_init(&insert_lk[i],NULL);
	}
}

bool partition_info::batch_insert(uint gid, uint num_objects, uint *pids){
	assert(num_objects<grid_capacity);
	pthread_mutex_lock(&insert_lk[gid%50]);
	memcpy(grids+(grid_capacity+1)*gid+1,pids,num_objects*sizeof(uint));
	*(grids+(grid_capacity+1)*gid) = num_objects;
	pthread_mutex_unlock(&insert_lk[gid%50]);
	return true;
}


bool partition_info::insert(uint gid, uint pid){
	pthread_mutex_lock(&insert_lk[gid%50]);
	uint cur_size = grids[(grid_capacity+1)*gid]++;
	grids[(grid_capacity+1)*gid+1+cur_size] = pid;
	pthread_mutex_unlock(&insert_lk[gid%50]);
	return true;
}

bool partition_info::check(uint gid, uint pid){
	assert(gid<num_grids);
	uint offset = 0;
	while(offset<get_grid_size(gid)){
		assert(num_checking_units<checking_units_capacity);
		checking_units[num_checking_units].pid = pid;
		checking_units[num_checking_units].gid = gid;
		checking_units[num_checking_units].offset = offset;
		num_checking_units++;
		offset += unit_size;
	}
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
		pinfo = new partition_info(num_grids, num_objects, config.grid_capacity, config.zone_capacity);
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
		size_t gid = grid->getgridid(points+pid);
		pinfo->check(gid,pid);

		size_t gid_code = grid->border_grids(points+pid,x_buffer,y_buffer);
		bool left = gid_code & 8;
		bool right = gid_code & 4;
		bool top = gid_code &2;
		bool bottom = gid_code &1;
		if(bottom&&left){
			pinfo->check(gid-grid->dimx-1,pid);
		}
		if(bottom){
			pinfo->check(gid-grid->dimx,pid);
		}
		if(bottom&&right){
			pinfo->check(gid-grid->dimx+1,pid);
		}
		if(left){
			pinfo->check(gid-1,pid);
		}
		if(right){
			pinfo->check(gid+1,pid);
		}
		if(top&&left){
			pinfo->check(gid+grid->dimx-1, pid);
		}
		if(top){
			pinfo->check(gid+grid->dimx,pid);
		}
		if(top&&right){
			pinfo->check(gid+grid->dimx+1, pid);
		}
	}

	logt("query",start);
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
		pinfo = new partition_info(num_grids, num_objects, config.grid_capacity, config.zone_capacity);
	}
	pinfo->points = points;

	pinfo->num_nodes = qtree->node_count();
	pinfo->schema = (QTSchema *)malloc(pinfo->num_nodes*sizeof(QTSchema));
	qtree->create_schema(pinfo->schema);

	logt("create schema",start);

	vector<QTNode *> leafs;
	qtree->get_leafs(leafs,false);
	for(QTNode *n:leafs){
		pinfo->batch_insert(n->node_id,n->object_index,n->objects);
	}
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

