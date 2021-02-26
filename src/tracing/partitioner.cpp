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

	// construct the QTree
	// todo parallelize it
	QTNode *qtree = new QTNode(mbr, config, points);
	for(uint pid=0;pid<num_objects;pid++){
		//log("%d",pid);
		qtree->insert(pid);
	}
	// set the ids and other stuff
	qtree->finalize();
	//qtree->print();


	// create and initialize the workbench
	workbench *bench = new workbench(config);

	bench->schema_counter = qtree->node_count();
	bench->grids_counter = qtree->leaf_count();
	assert(bench->schema_counter<bench->schema_capacity);
	cout<<bench->grids_counter<<" "<<bench->grids_capacity<<endl;
	assert(bench->grids_counter<bench->grids_capacity);

	bench->claim_space();

	// construct the schema with the QTree
	uint offset = 0;
	qtree->create_schema(bench->schema, offset);
	delete qtree;
	logt("partitioning schema is with %d grids",start,bench->grids_counter);
	return bench;
}
