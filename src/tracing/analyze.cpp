/*
 * analyze.cpp
 *
 *  Created on: Feb 25, 2021
 *      Author: teng
 */


#include "workbench.h"





void workbench::analyze_grids(){
	uint overflow = 0;
	uint max_one = 0;
	for(int i=0;i<schema_stack_capacity;i++){
		if(schema[i].type==LEAF){
			uint gid = schema[i].grid_id;
			uint gsize = grid_counter[gid];
			// todo increase the actuall capacity
			if(gsize>grid_capacity){
				overflow++;
			}
			if(max_one<gsize){
				max_one = gsize;
			}
		}
	}
	log("%d/%d overflow %d max",overflow,grids_stack_index,max_one);
}

void workbench::analyze_reaches(){
	uint *unit_count = new uint[config->num_objects];
	memset(unit_count,0,config->num_objects*sizeof(uint));
	uint min_bucket = 0;
	uint max_bucket = 0;
	uint total = 0;
	for(uint i=0;i<reaches_counter;i++){
		unit_count[reaches[i].pid1]++;
		unit_count[reaches[i].pid2]++;
	}
	uint max_one = 0;
	for(int i=0;i<config->num_objects;i++){
		if(unit_count[max_one]<unit_count[i]){
			max_one = i;
		}
	}

	uint *counter = new uint[unit_count[max_one]+1];
	memset(counter,0,(unit_count[max_one]+1)*sizeof(uint));

	for(int i=0;i<config->num_objects;i++){
		counter[unit_count[i]]++;
	}
	double cum_portion = 0;
	for(int i=0;i<=unit_count[max_one];i++){
		cum_portion += 1.0*counter[i]/config->num_objects;
		log("%d\t%d\t%f",i,counter[i],cum_portion);
	}
	delete counter;

	vector<Point *> max_reaches;
	for(uint i=0;i<reaches_counter;i++){
		if(reaches[i].pid2==max_one){
			max_reaches.push_back(points+reaches[i].pid1);
		}
		if(reaches[i].pid1==max_one){
			max_reaches.push_back(points+reaches[i].pid2);
		}
	}

	vector<Point *> all_points;
	vector<Point *> valid_points;
	Point *p1 = points + max_one;
	vector<uint> nodes;
	lookup_rec(schema, p1, 0, nodes, config->reach_distance, true);

	for(uint n:nodes){
		schema[n].mbr.print();
		uint gid = schema[n].grid_id;
		uint *cur_pid = get_grid(gid);
		for(uint i=0;i<get_grid_size(gid);i++){
			Point *p2 = points+cur_pid[i];
			if(p1==p2){
				continue;
			}
			all_points.push_back(p2);
			double dist = p1->distance(*p2,true);
			if(dist<=config->reach_distance){
				valid_points.push_back(p2);
			}
		}
	}


	p1->print();
	print_points(max_reaches);
	print_points(valid_points);
	print_points(all_points);

	log("point %d has %d contacts in result, %ld checked, %ld validated"
			,max_one,unit_count[max_one],all_points.size(), valid_points.size());
	max_reaches.clear();
	all_points.clear();
	valid_points.clear();
	delete []unit_count;
}


void workbench::analyze_meeting_buckets(){

}

void workbench::analyze_meetings(){

	uint *bucket_count = new uint[meeting_bucket_capacity];
	memset(bucket_count,0,meeting_bucket_capacity*sizeof(uint));

	uint min_bucket = 0;
	uint max_bucket = 0;
	uint total = 0;
	for(uint i=0;i<config->num_meeting_buckets;i++){
		total += meeting_buckets_counter[i];
		if(meeting_buckets_counter[min_bucket]>meeting_buckets_counter[i]){
			min_bucket = i;
		}
		if(meeting_buckets_counter[max_bucket]<meeting_buckets_counter[i]){
			max_bucket = i;
		}
		bucket_count[meeting_buckets_counter[i]]++;
	}
	log("total active meetings %d average %d bucket range: [%d, %d]",total,total/config->num_meeting_buckets,
			meeting_buckets_counter[min_bucket],meeting_buckets_counter[max_bucket]);
	double cum_portion = 0;
	for(int i=0;i<meeting_bucket_capacity;i++){
		if(bucket_count[i]>0){
			cum_portion += 1.0*bucket_count[i]/config->num_objects;
			log("%d\t%d\t%f",i,bucket_count[i],cum_portion);
		}
	}

	delete []bucket_count;
}

