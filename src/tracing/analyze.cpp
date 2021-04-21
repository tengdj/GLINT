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
	for(int i=0;i<grid_capacity;i++){
		gridc[i] = 0;
	}
	for(int i=0;i<schema_stack_capacity;i++){
		if(schema[i].type==LEAF){
			uint gid = schema[i].grid_id;
			uint gsize = grid_counter[gid];
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
	}

	if(pro.max_grid_num<total){
		pro.max_grid_num = total;
	}

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
	log("%d/%d overflow %d max",overflow,grids_stack_index,max_one);
	printf("%f\n",100.0*overflow/grids_stack_index);

}

void workbench::analyze_reaches(){

	uint *unit_count = new uint[config->num_objects];
	memset(unit_count,0,config->num_objects*sizeof(uint));
	uint min_bucket = 0;
	uint max_bucket = 0;
	uint total = 0;
	for(int i=0;i<config->num_meeting_buckets;i++){
		meeting_unit *bucket = meeting_buckets[current_bucket] + i*this->config->bucket_size;
		for(uint j=0;j<meeting_buckets_counter[current_bucket][i];j++){
			unit_count[bucket[j].pid1]++;
			unit_count[bucket[j].pid2]++;
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
	for(int i=0;i<config->num_meeting_buckets;i++){
		meeting_unit *bucket = meeting_buckets[current_bucket] + i*this->config->bucket_size;
		for(uint j=0;j<meeting_buckets_counter[current_bucket][i];j++){
			if(bucket[j].pid2==max_one){
				max_reaches.push_back(points+bucket[j].pid1);
			}
			if(bucket[j].pid1==max_one){
				max_reaches.push_back(points+bucket[j].pid2);
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


void workbench::analyze_meeting_buckets(){

	uint *bucket_count = new uint[config->bucket_size];
	memset(bucket_count,0,config->bucket_size*sizeof(uint));

	uint min_bucket = 0;
	uint max_bucket = 0;
	size_t total = 0;
	size_t overflow = 0;
	size_t overflow_count = 0;

	for(uint i=0;i<config->num_meeting_buckets;i++){
		total += meeting_buckets_counter[current_bucket][i];
		if(meeting_buckets_counter[current_bucket][min_bucket]>meeting_buckets_counter[current_bucket][i]){
			min_bucket = i;
		}
		if(meeting_buckets_counter[current_bucket][max_bucket]<meeting_buckets_counter[current_bucket][i]){
			max_bucket = i;
		}
		if(meeting_buckets_counter[current_bucket][i]<config->bucket_size){
			bucket_count[meeting_buckets_counter[current_bucket][i]]++;
		}else{
			overflow++;
			overflow_count += meeting_buckets_counter[current_bucket][i];
		}
	}
	log("total active meetings %d average %d bucket range: [%d, %d, %d]",
			total,total/config->num_meeting_buckets,
			meeting_buckets_counter[current_bucket][min_bucket],
			meeting_buckets_counter[current_bucket][max_bucket],
			config->bucket_size);
	double cum_portion = 0;
	uint vbuck = 0;
	for(int i=0;i<config->bucket_size;i++){
		cum_portion += 1.0*bucket_count[i]*i/total;
		if(config->analyze_meeting){
			printf("%d\t%d\t%.4f\n",i,bucket_count[i],cum_portion);
		}
		if(cum_portion>OVERFLOW_THRESHOLD&&vbuck==0){
			vbuck = i;
		}
	}

	if(pro.max_bucket_size<vbuck){
		pro.max_bucket_size = vbuck;
	}
	if(overflow>0){
		log("of\t%d\t%d\t%.4f",overflow,1.0*(overflow_count-overflow*config->bucket_size)/total);
	}
	double deviation = 0.0;
	double average = 1.0*total/config->num_meeting_buckets;
	for(uint i=0;i<config->num_meeting_buckets;i++){
		deviation += (meeting_buckets_counter[current_bucket][i]-average)*(meeting_buckets_counter[current_bucket][i]-average);
	}
	deviation = sqrt(deviation/config->num_meeting_buckets);
	//printf("%f %f %f\n",average, deviation, deviation/average);
	pro.meet_coefficient += deviation/average;
	pro.num_pairs += total;
	pro.num_meetings += this->meeting_counter;

	delete []bucket_count;
}

