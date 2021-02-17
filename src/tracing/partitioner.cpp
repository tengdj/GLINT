/*
 * partitioner.cpp
 *
 *  Created on: Feb 1, 2021
 *      Author: teng
 */

#include "trace.h"


partition_info::partition_info(configuration conf){
	config = conf;
	// each object can be checked with up to 5 grids, each with several checking units
	checking_units_capacity = config.num_objects*(config.grid_capacity/config.zone_capacity+1);
	for(int i=0;i<50;i++){
		pthread_mutex_init(&insert_lk[i],NULL);
	}
}

bool partition_info::batch_insert(uint gid, uint num_objects, uint *pids){
	assert(num_objects<config.grid_capacity);
	pthread_mutex_lock(&insert_lk[gid%50]);
	memcpy(grids+(config.grid_capacity+1)*gid+1,pids,num_objects*sizeof(uint));
	*(grids+(config.grid_capacity+1)*gid) = num_objects;
	pthread_mutex_unlock(&insert_lk[gid%50]);
	return true;
}


bool partition_info::insert(uint gid, uint pid){
	pthread_mutex_lock(&insert_lk[gid%50]);
	uint cur_size = grids[(config.grid_capacity+1)*gid]++;
	grids[(config.grid_capacity+1)*gid+1+cur_size] = pid;
	pthread_mutex_unlock(&insert_lk[gid%50]);
	return true;
}

bool partition_info::check(uint gid, uint pid){
	assert(gid<num_grids);
	pthread_mutex_lock(&insert_lk[0]);
	uint offset = 0;
	while(offset<get_grid_size(gid)){
		assert(num_checking_units<checking_units_capacity);
		checking_units[num_checking_units].pid = pid;
		checking_units[num_checking_units].gid = gid;
		checking_units[num_checking_units].offset = offset;
		num_checking_units++;
		offset += config.zone_capacity;
	}
	pthread_mutex_unlock(&insert_lk[0]);
	return true;
}

bool partition_info::batch_check(checking_unit *cu, uint num_cu){
	if(num_cu == 0){
		return false;
	}
	pthread_mutex_lock(&insert_lk[0]);
	assert(num_checking_units+num_cu<checking_units_capacity);
	memcpy(checking_units+num_checking_units,cu,sizeof(checking_unit)*num_cu);
	num_checking_units+= num_cu;
	pthread_mutex_unlock(&insert_lk[0]);
	return true;
}


void grid_partitioner::build_schema(Point *objects, size_t num_objects){
	if(grid){
		delete grid;
	}
	grid = new Grid(mbr, config.grid_width);
	size_t num_grids = grid->get_grid_num();
	if(!pinfo||pinfo->num_grids!=num_grids||pinfo->config.num_objects!=num_objects){
		if(pinfo){
			delete pinfo;
		}
		pinfo = new partition_info(config);
		pinfo->claim_space(num_grids);
	}
}

partition_info *grid_partitioner::partition(Point *points, size_t num_objects){
	// the schema is built
	assert(pinfo);
	struct timeval start = get_cur_time();

	pinfo->points = points;

	double x_buffer = config.reach_distance*degree_per_meter_longitude(grid->space.low[1]);
	double y_buffer = config.reach_distance*degree_per_meter_latitude;

	// assign each object to proper grids
	for(int pid=0;pid<num_objects;pid++){
		Point *p = points+pid;
		size_t gid = grid->getgridid(p);
		pinfo->insert(gid,pid);
	}
	logt("partition %ld objects into %ld grids",start,num_objects, pinfo->num_grids);

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


void qtree_partitioner::build_schema(Point *points, size_t num_objects){
	struct timeval start = get_cur_time();
	QTConfig qconfig;
	qconfig.min_width = config.reach_distance;
	qconfig.max_objects = config.grid_capacity;
	config.x_buffer = config.reach_distance*degree_per_meter_longitude(mbr.low[1]);
	config.y_buffer = config.reach_distance*degree_per_meter_latitude;
	qconfig.x_buffer = config.x_buffer;
	qconfig.y_buffer = config.y_buffer;
	QTNode *qtree = new QTNode(mbr, &qconfig);
	qconfig.points = points;

	for(uint pid=0;pid<num_objects;pid++){
		//log("%d",pid);
		qtree->insert(pid);
	}
	// set the ids and other stuff
	qtree->finalize();
	size_t num_grids = qtree->leaf_count();

	if(!pinfo||pinfo->num_grids!=num_grids||pinfo->config.num_objects!=num_objects){
		if(pinfo){
			delete pinfo;
		}
		pinfo = new partition_info(config);
		pinfo->claim_space(num_grids);
	}
	pinfo->num_nodes = qtree->node_count();
	pinfo->schema = qtree->create_schema();

	delete qtree;
	logt("partitioning schema is with %d grids",start,num_grids);
}

// single thread function for assigning objects into grids following certain schema
void *partition_unit(void *arg){
	query_context *qctx = (query_context *)arg;
	partition_info *pinfo = (partition_info *)qctx->target[0];
	QTSchema *schema = pinfo->schema;

	// pick one batch of point-grid pair for processing
	size_t start = 0;
	size_t end = 0;
	while(qctx->next_batch(start,end)){
		for(uint pid=start;pid<end;pid++){
			uint gid = 0;
			uint curoff = 0;
			Point *p = pinfo->points+pid;
			while(true){
				int loc = (p->y>pinfo->schema[curoff].mid_y)*2+(p->x>pinfo->schema[curoff].mid_x);
				// is leaf
				if((pinfo->schema[curoff].children[loc]&1)){
					gid = pinfo->schema[curoff].children[loc]>>1;
					break;
				}else{
					curoff = pinfo->schema[curoff].children[loc]>>1;
				}
			}
			pinfo->insert(gid, pid);
		}
	}
	return NULL;
}

void lookup(QTSchema *schema, Point *p, uint curoff, vector<uint> &gids, double x_buffer, double y_buffer){

	// could be possibly in multiple children with buffers enabled
	bool top = (p->y>schema[curoff].mid_y-y_buffer);
	bool bottom = (p->y<=schema[curoff].mid_y+y_buffer);
	bool left = (p->x<=schema[curoff].mid_x+x_buffer);
	bool right = (p->x>schema[curoff].mid_x-x_buffer);
	bool need_check[4] = {bottom&&left, bottom&&right, top&&left, top&&right};
	for(int i=0;i<4;i++){
		if(need_check[i]){
			if((schema[curoff].children[i]&1)){
				gids.push_back(schema[curoff].children[i]>>1);
			}else{
				lookup(schema, p, schema[curoff].children[i]>>1, gids, x_buffer, y_buffer);
			}
		}
	}
}

// single thread function for looking up the schema to generate point-grid pairs for processing
void *lookup_unit(void *arg){
	query_context *qctx = (query_context *)arg;
	partition_info *pinfo = (partition_info *)qctx->target[0];
	QTSchema *schema = pinfo->schema;

	// pick one batch of point-grid pair for processing
	size_t start = 0;
	size_t end = 0;
	vector<uint> gids;
	checking_unit *cubuffer = (checking_unit *)malloc(sizeof(checking_unit)*200);
	uint buffer_index = 0;
	while(qctx->next_batch(start,end)){
		for(uint pid=start;pid<end;pid++){
			Point *p = pinfo->points+pid;
			lookup(schema, p, 0, gids, qctx->config.x_buffer, qctx->config.y_buffer);

			for(uint gid:gids){
				assert(gid<pinfo->num_grids);
				uint offset = 0;
				while(offset<pinfo->get_grid_size(gid)){

					cubuffer[buffer_index].pid = pid;
					cubuffer[buffer_index].gid = gid;
					cubuffer[buffer_index].offset = offset;
					buffer_index++;
					if(buffer_index==200){
						pinfo->batch_check(cubuffer, buffer_index);
						buffer_index = 0;
					}
					offset += pinfo->config.zone_capacity;
				}
			}
			gids.clear();
		}
	}
	pinfo->batch_check(cubuffer, buffer_index);
	return NULL;
}



partition_info *qtree_partitioner::partition(Point *points, size_t num_objects){
	// the schema has to be built
	assert(pinfo);
	struct timeval start = get_cur_time();
	pinfo->reset();
	pinfo->points = points;
	if(config.gpu){
		return pinfo;
	}

	// partitioning current batch of objects with the existing schema
	pthread_t threads[config.num_threads];
	query_context qctx;
	qctx.config = config;
	qctx.num_objects = num_objects;
	qctx.target[0] = (void *)pinfo;

	for(int i=0;i<config.num_threads;i++){
		pthread_create(&threads[i], NULL, partition_unit, (void *)&qctx);
	}
	for(int i = 0; i < config.num_threads; i++ ){
		void *status;
		pthread_join(threads[i], &status);
	}
	logt("partition",start);

	// tree lookups
	qctx.reset();
	for(int i=0;i<config.num_threads;i++){
		pthread_create(&threads[i], NULL, lookup_unit, (void *)&qctx);
	}
	for(int i = 0; i < config.num_threads; i++ ){
		void *status;
		pthread_join(threads[i], &status);
	}
	logt("lookup",start);
	return pinfo;
}
