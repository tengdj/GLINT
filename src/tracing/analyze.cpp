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
	for(int gid=0;gid<grids_counter;gid++){
		uint gsize = grid_counter[gid];
		if(gsize>config->grid_capacity*1.5){
			overflow++;
		}
		if(max_one<gsize){
			max_one = gsize;
		}
	}
	log("%d/%d overflow %d max",overflow,grids_counter,max_one);

}

void workbench::analyze_checkings(){

}


void workbench::analyze_meeting_buckets(){

}

void workbench::analyze_meetings(){

	uint *unit_count = new uint[config->num_objects];
	memset(unit_count,0,config->num_objects*sizeof(uint));
	uint min_bucket = 0;
	uint max_bucket = 0;
	for(uint i=0;i<config->num_meeting_buckets;i++){
		if(meeting_buckets_counter[min_bucket]>meeting_buckets_counter[i]){
			min_bucket = i;
		}
		if(meeting_buckets_counter[max_bucket]<meeting_buckets_counter[i]){
			max_bucket = i;
		}
		meeting_unit *bucket = meeting_buckets+i*meeting_bucket_capacity;
		for(uint j=0;j<meeting_buckets_counter[i];j++){
			unit_count[bucket[j].pid1]++;
			unit_count[bucket[j].pid2]++;
		}
	}
	log("bucket range: [%d, %d]",meeting_buckets_counter[min_bucket],meeting_buckets_counter[max_bucket]);
	map<int, uint> connected;
	uint max_one = 0;
	for(int i=0;i<config->num_objects;i++){
		if(connected.find(unit_count[i])==connected.end()){
			connected[unit_count[i]] = 1;
		}else{
			connected[unit_count[i]]++;
		}
		if(unit_count[max_one]<unit_count[i]){
			max_one = i;
		}
	}
	log("%d contains max objects",max_one);
	double cum_portion = 0;
	for(auto a:connected){
		cum_portion += 1.0*a.second/config->num_objects;
		log("%d\t%d\t%f",a.first,a.second,cum_portion);
	}
	connected.clear();

	vector<Point *> all_points;
	vector<Point *> valid_points;
	Point *p1 = points + max_one;
	vector<uint> nodes;
	lookup_rec(schema, p1, 0, nodes, config->reach_distance, true);

	for(uint n:nodes){
		//schema[n].mbr.print();
		uint gid = schema[n].node_id;
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


	log("point %d has %d contacts in result, %ld checked, %ld validated"
			,max_one,unit_count[max_one],all_points.size(), valid_points.size());
	print_points(all_points);
	print_points(valid_points);
	p1->print();
	all_points.clear();
	valid_points.clear();
	delete []unit_count;

}

