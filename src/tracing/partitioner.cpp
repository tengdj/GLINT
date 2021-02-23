/*
 * partitioner.cpp
 *
 *  Created on: Feb 1, 2021
 *      Author: teng
 */

#include "trace.h"

workbench *partitioner::build_schema(Point *points, size_t num_objects){
	struct timeval start = get_cur_time();
	config->x_buffer = config->reach_distance*degree_per_meter_longitude(mbr.low[1]);
	config->y_buffer = config->reach_distance*degree_per_meter_latitude;
	QTNode *qtree = new QTNode(mbr, config, points);

	for(uint pid=0;pid<num_objects;pid++){
		//log("%d",pid);
		qtree->insert(pid);
	}
	// set the ids and other stuff
	qtree->finalize();
	size_t num_grids = qtree->leaf_count();
	//qtree->print();

	workbench *bench = new workbench(config);
	bench->claim_space(num_grids);

	bench->num_nodes = qtree->node_count();
	bench->schema = qtree->create_schema();

	delete qtree;
	logt("partitioning schema is with %d grids",start,num_grids);
	return bench;
}
