/*
 * tracing.h
 *
 *  Created on: Jan 19, 2021
 *      Author: teng
 */

#ifndef SRC_TRACING_TRACE_H_
#define SRC_TRACING_TRACE_H_

#include "../geometry/Map.h"
#include "../util/query_context.h"
#include "partitioner.h"
#include <map>

using namespace std;


/*
 *
 * the statistics of the trips parsed from the taxi data
 * each zone
 *
 * */
class ZoneStats{
public:
	int zoneid;
	int updated_round = 0;
	long count = 0;
	long duration = 0;
	double length = 0.0;
	double rate_sleep = 0.0;
	int max_sleep_time = 0;
	map<int,int> target_count;
	ZoneStats(int id){
		zoneid = id;
	}
	~ZoneStats(){
		target_count.clear();
	}
	double get_speed(){
		assert(length>0);
		assert(duration>0);
		return length/duration;
	}

};

class Event{
public:
	int timestamp;
	Point coordinate;
};

class Trip {
public:
	Event start;
	Event end;
	Trip(){};
	Trip(string str);
	void print_trip();
	int duration(){
		return end.timestamp-start.timestamp;
	}
	double speed(){
		return length()/duration();
	}
	double length(){
		return end.coordinate.distance(start.coordinate, true);
	}
	void resize(int max_duration);
};

class trace_generator{
	Grid *grid = NULL;
	vector<ZoneStats *> zones;
	vector<ZoneStats *> ordered_zones;
	ZoneStats *total = NULL;
public:

    Map *map = NULL;
    configuration config;

	// construct with some parameters
	trace_generator(configuration &c, Map *m){
		config = c;
		assert(config.num_threads>0);
		map = m;
		grid = new Grid(*m->getMBR(),config.grid_width);
		zones.resize(grid->get_grid_num());
		ordered_zones.resize(grid->get_grid_num());
		for(int i=0;i<zones.size();i++){
			zones[i] = new ZoneStats(i);
			ordered_zones[i] = zones[i];
		}
	}
	~trace_generator(){
		map = NULL;
		for(ZoneStats *z:zones){
			delete z;
		}
		zones.clear();
		if(grid){
			delete grid;
		}
		if(total){
			delete total;
		}
		ordered_zones.clear();
	}
	// generate the destination with the given source point
	Trip *next_trip(Trip *former=NULL);

	void analyze_trips(const char *path, int limit = 2147483647);
	Point *generate_trace();
	// generate a trace with given duration
	vector<Point *> get_trace(Map *mymap = NULL);

};


class tracer{
	// the statistic for the data set
	Point *trace = NULL;
	bool owned_trace = false;
	// for query
	configuration config;
	partitioner *part = NULL;
	uint *result = NULL;
public:
	box mbr;
	tracer(configuration &conf, box &b, Point *t){
		trace = t;
		mbr = b;
		config = conf;
		if(config.method == QTREE){
			part = new qtree_partitioner(mbr,config);
		}else if(config.method == FIX_GRID){
			part = new grid_partitioner(mbr,config);
		}
		result = new uint[config.num_objects];
	}
	tracer(configuration &conf){
		config = conf;
		loadFrom(conf.trace_path.c_str());
		if(config.method == QTREE){
			part = new qtree_partitioner(mbr,config);
		}else if(config.method == FIX_GRID){
			part = new grid_partitioner(mbr,config);
		}
		result = new uint[config.num_objects];
	};
	~tracer(){
		if(owned_trace){
			free(trace);
		}
		if(part){
			delete part;
		}
		delete []result;
	}
	void process();
	void dumpTo(const char *path) {
		struct timeval start_time = get_cur_time();
		ofstream wf(path, ios::out|ios::binary|ios::trunc);
		wf.write((char *)&config.num_objects, sizeof(config.num_objects));
		wf.write((char *)&config.duration, sizeof(config.duration));
		wf.write((char *)&mbr, sizeof(mbr));
		size_t num_points = config.duration*config.num_objects;
		wf.write((char *)trace, sizeof(Point)*num_points);
		wf.close();
		logt("dumped to %s",start_time,path);
	}

	void loadFrom(const char *path) {

		int total_num_objects;
		int total_duration;
		struct timeval start_time = get_cur_time();
		ifstream in(path, ios::in | ios::binary);
		in.read((char *)&total_num_objects, sizeof(total_num_objects));
		in.read((char *)&total_duration, sizeof(total_duration));
		in.read((char *)&mbr, sizeof(mbr));
		mbr.to_squre(true);
		assert(config.duration*config.num_objects<=total_duration*total_num_objects);

		trace = (Point *)malloc(config.duration*config.num_objects*sizeof(Point));
		for(int i=0;i<config.duration;i++){
			in.read((char *)(trace+i*config.num_objects), config.num_objects*sizeof(Point));
			if(total_num_objects>config.num_objects){
				in.seekg((total_num_objects-config.num_objects)*sizeof(Point), ios_base::cur);
			}
		}
		in.close();
		logt("loaded %d objects last for %d seconds from %s",start_time, config.num_objects, config.duration, path);
		owned_trace = true;
	}

	void print_trace(double sample_rate){
		print_points(trace,config.num_objects,sample_rate);
	}
	Point *get_trace(){
		return trace;
	}
};



#endif /* SRC_TRACING_TRACE_H_ */
