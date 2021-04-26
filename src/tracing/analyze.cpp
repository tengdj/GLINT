/*
 * analyze.cpp
 *
 *  Created on: Feb 25, 2021
 *      Author: teng
 */


#include "workbench.h"

#define OVERFLOW_THRESHOLD 0.997

void workbench::analyze_grids(){
	uint overflow = 0;
	uint max_one = 0;
	uint *gridc = new uint[grid_capacity];
	uint total = 0;
	uint total_schema = 0;
	for(int i=0;i<grid_capacity;i++){
		gridc[i] = 0;
	}

	double mean = config->num_objects/this->grids_stack_index;
	double dev = 0.0;

	for(int i=0;i<schema_stack_capacity;i++){
		if(schema[i].type==LEAF){
			uint gid = schema[i].grid_id;
			uint gsize = grid_counter[gid];
			dev += (gsize-mean)*(gsize-mean);
			// todo increase the actual capacity
			if(gsize>grid_capacity){
				overflow++;
			}
			gridc[gsize>=grid_capacity?(grid_capacity-1):gsize]++;
			total++;
			if(max_one<gsize){
				max_one = gsize;
			}
		}
		total_schema += (schema[i].type!=INVALID);
	}

	pro.max_schema_num = max(pro.max_schema_num, total_schema);
	pro.max_grid_num = max(pro.max_grid_num, total);

	double cum = 0;
	for(int i=0;i<grid_capacity;i++){
		cum += 1.0*gridc[i]/total;
		if(cum>OVERFLOW_THRESHOLD){
			if(pro.max_grid_size<i){
				pro.max_grid_size = i;
			}
			break;
		}
	}
	pro.grid_count += grids_stack_index;
	pro.grid_overflow += overflow;
	pro.grid_overflow_list.push_back(100.0*overflow/grids_stack_index);
	pro.grid_deviation_list.push_back(sqrt(dev/grids_stack_index));
	log("%d/%d overflow %d max",overflow,grids_stack_index,max_one);
}

void workbench::analyze_reaches(){

	uint *unit_count = new uint[config->num_objects];
	memset(unit_count,0,config->num_objects*sizeof(uint));
	uint min_bucket = 0;
	uint max_bucket = 0;
	uint total = 0;
	for(size_t i=0;i<config->num_meeting_buckets;i++){
		if(!meeting_buckets[i].isEmpty()){
			pair<uint, uint> pids = InverseCantorPairing1(meeting_buckets[i].key);
			unit_count[pids.first]++;
			unit_count[pids.second]++;
		}
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
		log("%d\t%d\t%.3f",i,counter[i],cum_portion);
	}
	delete counter;

	vector<Point *> max_reaches;
	for(size_t i=0;i<config->num_meeting_buckets;i++){
		if(!meeting_buckets[i].isEmpty()){
			pair<uint, uint> pids = InverseCantorPairing1(meeting_buckets[i].key);
			if(pids.second==max_one){
				max_reaches.push_back(points+pids.first);
			}
			if(pids.first==max_one){
				max_reaches.push_back(points+pids.second);
			}
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

