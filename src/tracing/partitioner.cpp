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

	// make the capacity of zones half more than the grid number
	capacity = num_grids+num_grids/2;
	buffer_zones = new uint[(zone_size+2)*capacity];
	cur_zone = new uint[num_grids];
	grid_checkings = new uint[2*10*num_objects];

	clear();
	pthread_mutex_init(&lk,NULL);
	for(int i=0;i<50;i++){
		pthread_mutex_init(&insert_lk[i],NULL);
	}
	log("zone size: %d",zone_size);
}

// resize the zone buffer, should barely called
bool partition_info::resize(size_t newcapacity){
	if(newcapacity<capacity){
		return false;
	}
	uint *npart = new uint[(zone_size+2)*newcapacity];
	memcpy(npart,buffer_zones,(zone_size+2)*capacity*sizeof(uint));
	delete []buffer_zones;
	buffer_zones = npart;
	capacity = newcapacity;
	return true;
}


int partition_info::batch_insert(uint gid, uint num_objects, uint *pids){
	pthread_mutex_lock(&insert_lk[gid%50]);
	// the batch_insert can only be called once for a grid that is not loaded
	assert(cur_zone[gid]==0);
	uint inserted = 0;
	int num_zones = 0;
	while(inserted<num_objects){
		pthread_mutex_lock(&lk);
		// enlarge the buffer zone by 50% if the capacity is reached
		if(cur_free_zone==capacity){
			enlarge();
		}
		uint new_zid = cur_free_zone++;
		pthread_mutex_unlock(&lk);

		// link to the previous zone offset
		set_prev_zoneid(new_zid, cur_zone[gid]);

		cur_zone[gid] = new_zid;

		uint cur_inserted = min(num_objects-inserted, (uint)zone_size);
		memcpy(buffer_zones+(zone_size+2)*cur_zone[gid]+2,pids+inserted,cur_inserted*sizeof(uint));
		set_zone_size(cur_zone[gid], cur_inserted);

		buffer_zones[(zone_size+2)*cur_zone[gid]+1] = cur_inserted;
		inserted += cur_inserted;
		num_zones++;
	}
	pthread_mutex_unlock(&insert_lk[gid%50]);
	return num_zones;
}


bool partition_info::insert(uint gid, uint pid){
	pthread_mutex_lock(&insert_lk[gid%50]);
	uint cur_size = get_zone_size(cur_zone[gid])%zone_size;
	// one buffer zone is full or no zone is assigned, locate next buffer zone
	if(cur_size==0){
		pthread_mutex_lock(&lk);
		// enlarge the buffer zone by 50% if the capacity is reached
		if(cur_free_zone==capacity){
			enlarge();
		}
		uint new_zid = cur_free_zone++;
		pthread_mutex_unlock(&lk);

		// link to the previous zone offset
		set_prev_zoneid(new_zid, cur_zone[gid]);
		// the size for the newly allocated zone is 0
		set_zone_size(new_zid, 0);
		cur_zone[gid] = new_zid;
	}
	// the first number in the buffer zone links to the previous zone (0 if no one)
	// the second number in the buffer zone is the number of objects in current zone
	buffer_zones[(zone_size+2)*cur_zone[gid]+2+cur_size] = pid;
	// increase size by one
	buffer_zones[(zone_size+2)*cur_zone[gid]+1]++;
	//printf("%d\t%d\t%d\t%d\n",gid,cur_zone[gid],buffer_zones[(zone_size+2)*cur_zone[gid]],buffer_zones[(zone_size+2)*cur_zone[gid]+1]);

	pthread_mutex_unlock(&insert_lk[gid%50]);
	return true;
}

bool partition_info::check(uint gid, uint pid){
	assert(gid<num_grids);
	uint zid = cur_zone[gid];
	while(zid){
		assert(num_grid_checkings<10*num_objects);
		grid_checkings[2*num_grid_checkings] = pid;
		grid_checkings[2*num_grid_checkings+1] = zid;
		num_grid_checkings++;
		// the first number in this zone point to the previous zone
		zid = buffer_zones[(zone_size+2)*zid];
	};
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
		size_t zone_size = config.grid_capacity;
		pinfo = new partition_info(num_grids, num_objects, zone_size);
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

	uint total = 0;
	for(int i=0;i<grid->get_grid_num();i++){
		uint size = pinfo->get_grid_size(i);
		uint ct = pinfo->get_zone_count(i);
		total += size*ct;
	}

	cout<<total<<endl;
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
		pinfo = new partition_info(num_grids, num_objects, config.grid_capacity);
	}
	pinfo->points = points;

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

