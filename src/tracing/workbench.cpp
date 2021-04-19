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
	current_bucket = bench->current_bucket;
	cur_time = bench->cur_time;
}

workbench::workbench(configuration *conf){
	config = conf;

	// setup the capacity of each container
	grid_capacity = 2*config->grid_capacity;
	// each grid contains averagely grid_capacity/2 objects, times 4 for enough space
	grids_stack_capacity = 4*max((uint)1, config->num_objects/config->grid_capacity);

	// the number of all QTree Nodes
	schema_stack_capacity = 2*grids_stack_capacity;

	global_stack_capacity = 2*config->num_objects;

	grid_check_capacity = 2*config->num_objects*(max(config->grid_capacity/config->zone_capacity,(uint)1));

	meeting_bucket_capacity = max((uint)20, 2*config->num_objects/config->num_meeting_buckets);

	//	meeting_capacity = config->num_objects/2;
	meeting_capacity = 100;

	insert_lk = new pthread_mutex_t[MAX_LOCK_NUM];
	for(int i=0;i<MAX_LOCK_NUM;i++){
		pthread_mutex_init(&insert_lk[i],NULL);
	}
	for(int i=0;i<100;i++){
		data[i] = NULL;
		data_size[i] = 0;
	}
}

void workbench::clear(){
	for(int i=0;i<100;i++){
		if(data[i]!=NULL){
			free(data[i]);
			data[i] = NULL;
			data_size[i] = 0;
		}
		data_index = 0;
	}
	delete insert_lk;
}


void *workbench::allocate(size_t size){
	lock();
	uint cur_idx = data_index++;
	unlock();
	data[cur_idx] = malloc(size);
	data_size[cur_idx] = size;
	return data[cur_idx];
}

void workbench::claim_space(){

	size_t size = 0;

	size = grid_capacity*grids_stack_capacity*sizeof(uint);
	grids = (uint *)allocate(size);
	log("\t%.2f MB\tgrids",size/1024.0/1024.0);

	size = grids_stack_capacity*sizeof(uint);
	grid_counter = (uint *)allocate(size);
	//log("\t%.2f MB\tgrid counter",size/1024.0/1024.0);

	size = grids_stack_capacity*sizeof(uint);
	grids_stack = (uint *)allocate(size);
	//log("\t%.2f MB\tgrids stack",size/1024.0/1024.0);
	for(int i=0;i<grids_stack_capacity;i++){
		grids_stack[i] = i;
	}

	size = schema_stack_capacity*sizeof(QTSchema);
	schema = (QTSchema*)allocate(size);
	log("\t%.2f MB\tschema",size/1024.0/1024.0);

	size = schema_stack_capacity*sizeof(uint);
	schema_stack = (uint *)allocate(size);
	//log("\t%.2f MB\tschema stack",size/1024.0/1024.0);
	for(int i=0;i<schema_stack_capacity;i++){
		schema_stack[i] = i;
		schema[i].type = INVALID;
	}

	size = grid_check_capacity*sizeof(checking_unit);
	grid_check = (checking_unit *)allocate(size);
	log("\t%.2f MB\tchecking units",size/1024.0/1024.0);

	size = config->num_meeting_buckets*meeting_bucket_capacity*sizeof(meeting_unit);
	meeting_buckets[0] = (meeting_unit *)allocate(size);
	meeting_buckets[1] = (meeting_unit *)allocate(size);
	log("\t%.2f MB\tmeeting bucket space",2*size/1024.0/1024.0);

	size = config->num_meeting_buckets*sizeof(uint);
	meeting_buckets_counter[0] = (uint *)allocate(size);
	memset(meeting_buckets_counter[0],0,config->num_meeting_buckets*sizeof(uint));
	meeting_buckets_counter[1] = (uint *)allocate(size);
	memset(meeting_buckets_counter[1],0,config->num_meeting_buckets*sizeof(uint));
	//log("\t%.2f MB  \tmeeting bucket counter space",2*size/1024.0/1024.0);

	size = meeting_capacity*sizeof(meeting_unit);
	meetings = (meeting_unit *)allocate(size);
	log("\t%.2f MB\tmeeting space",size/1024.0/1024.0);

	size = global_stack_capacity*2*sizeof(uint);
	global_stack[0] = (uint *)allocate(size);
	global_stack[1] = (uint *)allocate(size);
	log("\t%.2f MB\tstack space",2*size/1024.0/1024.0);
}


void workbench::print_profile(){
	printf("memory space:\n");
	printf("\tgrid buffer:\t%ld MB\n",pro.max_grid_num*pro.max_grid_size*sizeof(uint)/1024/1024);
	printf("\trefine list:\t%ld MB\n",pro.max_filter_size*sizeof(checking_unit)/1024/1024);
	printf("\tmeeting table:\t%ld MB\n",2*pro.max_bucket_size*config->num_meeting_buckets*sizeof(meeting_unit)/1024/1024);
	printf("\tstack size:\t%ld MB\n",2*pro.max_stak_size*2*sizeof(uint)/1024/1024);

	if(pro.rounds>0){
		printf("time cost:\n");
		printf("\tcopy data:\t%.2f\n",pro.copy_time/pro.rounds);
		printf("\tfiltering:\t%.2f\n",pro.filter_time/pro.rounds);
		printf("\trefinement:\t%.2f\n",pro.refine_time/pro.rounds);
		printf("\tupdate meets:\t%.2f\n",pro.meeting_update_time/pro.rounds);
		printf("\tupdate index:\t%.2f\n",pro.index_update_time/pro.rounds);
		double overall = pro.copy_time+pro.filter_time+pro.refine_time+pro.meeting_update_time+pro.index_update_time;
		printf("\toverall:\t%.2f\n",overall/pro.rounds);
	}
}

