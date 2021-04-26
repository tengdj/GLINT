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
	cur_time = bench->cur_time;
}

workbench::workbench(configuration *conf){

	config = conf;

	// setup the capacity of each container
	grid_capacity = config->grid_amplify*config->grid_capacity;
	// each grid contains averagely grid_capacity/2 objects, times 4 for enough space
	grids_stack_capacity = 4*max((uint)1, config->num_objects/config->grid_capacity);

	// the number of all QTree Nodes
	schema_stack_capacity = 1.6*grids_stack_capacity;

	tmp_space_capacity = config->num_objects;

	filter_list_capacity = config->num_objects;

	grid_check_capacity = config->refine_size*config->num_objects;

	meeting_capacity = config->num_objects/4;
	//meeting_capacity = 100;

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

	size = config->num_meeting_buckets*sizeof(meeting_unit);
	meeting_buckets = (meeting_unit *)allocate(size);
	log("\t%.2f MB\tmeeting bucket space",size/1024.0/1024.0);

	size = meeting_capacity*sizeof(meeting_unit);
	meetings = (meeting_unit *)allocate(size);
	log("\t%.2f MB\tmeeting space",size/1024.0/1024.0);

#pragma omp parallel for
	for(size_t i=0;i<config->num_meeting_buckets;i++){
		meeting_buckets[i].key = ULL_MAX;
	}

}


void workbench::print_profile(){

	fprintf(stderr,"memory space:\n");
	fprintf(stderr,"\tgrid buffer:\t%ld MB\n",pro.max_grid_num*pro.max_grid_size*sizeof(uint)/1024/1024);
	fprintf(stderr,"\tschema list:\t%ld MB\n",pro.max_schema_num*sizeof(QTSchema)/1024/1024);

	fprintf(stderr,"\tfilter list:\t%ld MB\n",pro.max_filter_size*2*sizeof(uint)/1024/1024);
	fprintf(stderr,"\trefine list:\t%ld MB\n",pro.max_refine_size*sizeof(checking_unit)/1024/1024);
	fprintf(stderr,"\tmeeting table:\t%ld MB\n",pro.max_bucket_num*sizeof(meeting_unit)/1024/1024);
	fprintf(stderr,"\tstack size:\t%ld MB\n",tmp_space_capacity*sizeof(meeting_unit)/1024/1024);

	if(pro.rounds>0){
		fprintf(stderr,"time cost:\n");
		fprintf(stderr,"\tcopy data:\t%.2f\n",pro.copy_time/pro.rounds);
		fprintf(stderr,"\tfiltering:\t%.2f\n",pro.filter_time/pro.rounds);
		fprintf(stderr,"\trefinement:\t%.2f\n",pro.refine_time/pro.rounds);
		fprintf(stderr,"\tupdate meets:\t%.2f\n",pro.meeting_identify_time/pro.rounds);
		fprintf(stderr,"\tupdate index:\t%.2f\n",pro.index_update_time/pro.rounds);
		fprintf(stderr,"\toverall:\t%.2f\n",(pro.copy_time+pro.filter_time+pro.refine_time+pro.meeting_identify_time+pro.index_update_time)/pro.rounds);


		fprintf(stderr,"statistics:\n");
		fprintf(stderr,"\tnum pairs:\t%.2f \n",2.0*(pro.num_pairs/pro.rounds)/config->num_objects);
		fprintf(stderr,"\tnum meetings:\t%ld \n",pro.num_meetings/pro.rounds);
		fprintf(stderr,"\tusage rate:\t%.2f%% (%ld/%ld)\n",100.0*(pro.max_bucket_num)/config->num_meeting_buckets,pro.max_bucket_num,config->num_meeting_buckets);
		fprintf(stderr,"\t80 usage:\t%ld\n",(size_t)(pro.max_bucket_num/0.8));
		fprintf(stderr,"\toverflow:\t%.4f\n",100.0*pro.grid_overflow/pro.grid_count);
	}

	printf("grid_buffer,schema,filter_list,refine_list,meeting_table,stack_size,");
	printf("copy,filtering,refinement,identify,update_index,");
	printf("num pairs,num meetings,overflow\n");

	printf("%ld,%ld,%ld,%ld,%ld,%ld,",pro.max_grid_num*pro.max_grid_size*sizeof(uint)/1024/1024
													   ,pro.max_schema_num*sizeof(QTSchema)/1024/1024
													   ,pro.max_filter_size*2*sizeof(uint)/1024/1024
													   ,pro.max_refine_size*sizeof(checking_unit)/1024/1024
													   ,pro.max_bucket_num*sizeof(meeting_unit)/1024/1024
													   ,tmp_space_capacity*sizeof(meeting_unit)/1024/1024);

	printf("%.2f,%.2f,%.2f,%.2f,%.2f,",pro.copy_time/pro.rounds
										   ,pro.filter_time/pro.rounds
										   ,pro.refine_time/pro.rounds
										   ,pro.meeting_identify_time/pro.rounds
										   ,pro.index_update_time/pro.rounds);

	printf("%.2f,%ld,%.4f\n",2.0*(pro.num_pairs/pro.rounds)/config->num_objects
							  ,pro.num_meetings/pro.rounds
							  ,100.0*pro.grid_overflow/pro.grid_count);


	printf("overflow rate:\n");
	for(double o:pro.grid_overflow_list){
		printf("%f\n", o);
	}

	printf("deviation:\n");
	for(double o:pro.grid_deviation_list){
		printf("%f\n", o);
	}

	// bucket number
	//printf("%ld\t%.2f\t%.4f\n",2*pro.max_bucket_size*config->num_meeting_buckets*sizeof(meeting_unit)/1024/1024,pro.meeting_update_time/pro.rounds,pro.meet_coefficient/pro.rounds);

	// grid size
	//printf("%.2f\t%.2f\t%.2f\t%ld\t%.4f\n",pro.filter_time/pro.rounds,pro.refine_time/pro.rounds,pro.index_update_time/pro.rounds,pro.
	//		max_filter_size*sizeof(checking_unit)/1024/1024,100.0*pro.grid_overflow/pro.grid_count);

	// minimum distance
//	printf("%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n",pro.filter_time/pro.rounds,pro.refine_time/pro.rounds,pro.index_update_time/pro.rounds,
//			pro.meeting_update_time/pro.rounds,overall/pro.rounds,2.0*(pro.num_pairs/pro.rounds)/config->num_objects);

	// minimum duration
	//printf("%.2f\t%.2f\t%ld\n",pro.copy_time/pro.rounds,pro.meeting_update_time/pro.rounds,pro.num_meetings/pro.rounds);

}

