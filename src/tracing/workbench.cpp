/*
 * workbench.cpp
 *
 *  Created on: Feb 18, 2021
 *      Author: teng
 */

#include "workbench.h"


workbench::workbench(workbench *bench):workbench(bench->config){
	grids_stack_index = bench->grids_stack_index;
	schema_stack_index = bench->schema_stack_index;
}

workbench::workbench(configuration *conf){
	config = conf;

	// setup the capacity of each container
	grid_capacity = 10*config->grid_capacity;
	// each grid contains averagely grid_capacity/2 objects, times 3 for enough space
	grids_stack_capacity = 3*max((uint)1, config->num_objects/config->grid_capacity);
	// the number of all QTree Nodes
	schema_stack_capacity = 2*grids_stack_capacity;
	lookup_stack_capacity = 2*config->num_objects;
	reaches_capacity = 10*config->num_objects;
	meeting_capacity = 10*config->num_objects;
	meeting_bucket_capacity = max((uint)20, reaches_capacity/config->num_meeting_buckets);
	grid_check_capacity = config->num_objects*(config->grid_capacity/config->zone_capacity+1);

	for(int i=0;i<50;i++){
		pthread_mutex_init(&insert_lk[i],NULL);
	}
}

void workbench::clear(){
	if(grids){
		delete []grids;
	}
	if(grid_counter){
		delete []grid_counter;
	}
	if(grid_check){
		delete []grid_check;
	}
	if(schema){
		delete []schema;
	}
	if(reaches){
		delete []reaches;
	}
	if(meeting_buckets){
		delete []meeting_buckets;
	}
	if(meeting_buckets_counter){
		delete []meeting_buckets_counter;
	}
	if(meeting_buckets_counter_tmp){
		delete []meeting_buckets_counter_tmp;
	}
	if(meetings){
		delete []meetings;
	}
	if(lookup_stack[0]){
		delete lookup_stack[0];
	}
	if(lookup_stack[1]){
		delete lookup_stack[1];
	}
}


void workbench::claim_space(){

	struct timeval start = get_cur_time();

	double total_size = 0;
	double tmp_size = 0;

	grids = new uint[grid_capacity*grids_stack_capacity];
	tmp_size = grid_capacity*grids_stack_capacity*sizeof(uint)/1024.0/1024.0;
	log("\t%.2f MB\tgrids",tmp_size);
	total_size += tmp_size;

	grid_counter = new uint[grids_stack_capacity];
	tmp_size = grids_stack_capacity*sizeof(uint)/1024.0/1024.0;
	log("\t%.2f MB\tgrid counter",tmp_size);
	total_size += tmp_size;

	grids_stack = new uint[grids_stack_capacity];
	tmp_size = grids_stack_capacity*sizeof(uint)/1024.0/1024.0;
	log("\t%.2f MB\tgrids stack",tmp_size);
	total_size += tmp_size;

	schema = new QTSchema[2*schema_stack_capacity];
	tmp_size = 2*grids_stack_capacity*sizeof(QTSchema)/1024.0/1024.0;
	log("\t%.2f MB\tschema",tmp_size);
	total_size += tmp_size;

	schema_stack = new uint[schema_stack_capacity];
	tmp_size = schema_stack_capacity*sizeof(uint)/1024.0/1024.0;
	log("\t%.2f MB\tschema stack",tmp_size);
	total_size += tmp_size;

	grid_check = new checking_unit[grid_check_capacity];
	tmp_size = grid_check_capacity*sizeof(checking_unit)/1024.0/1024.0;
	log("\t%.2f MB\tchecking units",tmp_size);
	total_size += tmp_size;

	reaches = new reach_unit[reaches_capacity];
	tmp_size = reaches_capacity*sizeof(reach_unit)/1024.0/1024.0;
	log("\t%.2f MB\treach space",tmp_size);
	total_size += tmp_size;

	meeting_buckets = new meeting_unit[config->num_meeting_buckets*meeting_bucket_capacity];
	tmp_size = config->num_meeting_buckets*meeting_bucket_capacity*sizeof(meeting_unit)/1024.0/1024.0;
	log("\t%.2f MB\tmeeting bucket space",tmp_size);
	total_size += tmp_size;

	meeting_buckets_counter = new uint[config->num_meeting_buckets];
	memset(meeting_buckets_counter,0,config->num_meeting_buckets*sizeof(uint));
	meeting_buckets_counter_tmp = new uint[config->num_meeting_buckets];
	memset(meeting_buckets_counter_tmp,0,config->num_meeting_buckets*sizeof(uint));
	tmp_size = 2*config->num_meeting_buckets*sizeof(uint)/1024.0/1024.0;
	log("\t%.2f MB  \tmeeting bucket counter space",tmp_size);
	total_size += tmp_size;

	meetings = new meeting_unit[meeting_capacity];
	tmp_size = meeting_capacity*sizeof(meeting_unit)/1024.0/1024.0;
	log("\t%.2f MB\tmeeting space",tmp_size);
	total_size += tmp_size;

	lookup_stack[0] = new uint[2*lookup_stack_capacity];
	lookup_stack[1] = new uint[2*lookup_stack_capacity];
	tmp_size = 2*lookup_stack_capacity*2*sizeof(uint)/1024.0/1024.0;
	log("\t%.2f MB\tstack space",tmp_size);
	total_size += tmp_size;

	logt("%.2f MB memory space is claimed",start,total_size);
}
