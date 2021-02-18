/*
 * workbench.cpp
 *
 *  Created on: Feb 18, 2021
 *      Author: teng
 */

#include "workbench.h"


workbench::workbench(configuration conf){
	config = conf;

	meeting_capacity = config.num_objects;
	// todo: figure out a way to reduce the total amount
	checking_units_capacity = config.num_objects*(config.grid_capacity/config.zone_capacity+1);
	for(int i=0;i<50;i++){
		pthread_mutex_init(&insert_lk[i],NULL);
	}
}

workbench::~workbench(){
	if(grids){
		delete []grids;
	}
	if(checking_units){
		delete []checking_units;
	}
	if(schema){
		delete []schema;
	}
	if(meetings){
		delete []meetings;
	}
}


void workbench::claim_space(uint ng){
		assert(ng>0);
		if(num_grids != ng){
			if(grids){
				delete []grids;
			}
			num_grids = ng;
			grids = new uint[(config.grid_capacity+1)*num_grids];
		}

		if(!checking_units){
			checking_units = new checking_unit[checking_units_capacity];
		}
		if(!meetings){
			meetings = new meeting_unit[meeting_capacity];
		}
	}

bool workbench::batch_insert(uint gid, uint num_objects, uint *pids){
	assert(num_objects<config.grid_capacity);
	pthread_mutex_lock(&insert_lk[gid%50]);
	memcpy(grids+(config.grid_capacity+1)*gid+1,pids,num_objects*sizeof(uint));
	*(grids+(config.grid_capacity+1)*gid) = num_objects;
	pthread_mutex_unlock(&insert_lk[gid%50]);
	return true;
}


bool workbench::insert(uint gid, uint pid){
	pthread_mutex_lock(&insert_lk[gid%50]);
	uint cur_size = grids[(config.grid_capacity+1)*gid]++;
	grids[(config.grid_capacity+1)*gid+1+cur_size] = pid;
	pthread_mutex_unlock(&insert_lk[gid%50]);
	return true;
}

bool workbench::check(uint gid, uint pid){
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

bool workbench::batch_check(checking_unit *cu, uint num_cu){
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


void workbench::analyze_grids(){

}

void workbench::analyze_checkings(){
	uint *unit_count = new uint[config.num_objects];
	memset(unit_count,0,config.num_objects*sizeof(uint));
	for(uint pairid=0;pairid<num_checking_units;pairid++){
		unit_count[checking_units[pairid].pid]++;
	}
	uint max_one = 0;
	for(int i=0;i<config.num_objects;i++){
		if(unit_count[max_one]<unit_count[i]){
			max_one = i;
		}
	}
	cout<<max_one<<" "<<unit_count[max_one]<<endl;
	delete []unit_count;
}


void workbench::analyze_meetings(){

	/*
	 *
	 * some statistics printing for debuging only
	 *
	 * */

	uint *unit_count = new uint[config.num_objects];
	memset(unit_count,0,config.num_objects*sizeof(uint));
	for(uint mid=0;mid<num_meeting;mid++){
		unit_count[meetings[mid].pid1]++;
	}
	map<int, uint> connected;
	uint max_one = 0;
	for(int i=0;i<config.num_objects;i++){
		if(connected.find(unit_count[i])==connected.end()){
			connected[unit_count[i]] = 1;
		}else{
			connected[unit_count[i]]++;
		}
		if(unit_count[max_one]<unit_count[i]){
			max_one = i;
		}
	}
	double cum_portion = 0;
	for(auto a:connected){
		cum_portion += 1.0*a.second/config.num_objects;
		printf("%d\t%d\t%f\n",a.first,a.second,cum_portion);
	}
	connected.clear();

	vector<Point *> all_points;
	vector<Point *> valid_points;
	Point *p1 = points + max_one;
	for(uint pairid=0;pairid<num_checking_units;pairid++){
		if(checking_units[pairid].pid==max_one&&checking_units[pairid].offset==0){
			uint *cur_pid = get_grid(checking_units[pairid].gid);
			for(uint i=0;i<get_grid_size(checking_units[pairid].gid);i++){
				Point *p2 = points+cur_pid[i];
				if(p1==p2){
					continue;
				}
				all_points.push_back(p2);
				double dist = p1->distance(*p2,true);
				if(dist<config.reach_distance){
					valid_points.push_back(p2);
				}
			}
		}
	}

	print_points(all_points);
	print_points(valid_points);
	p1->print();
	printf("point %d has %d contacts in result, %ld checked, %ld validated\n"
			,max_one,unit_count[max_one],all_points.size(), valid_points.size());
	all_points.clear();
	valid_points.clear();
	delete []unit_count;

}



