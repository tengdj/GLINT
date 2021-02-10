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
	double length(bool geography=true){
		return end.coordinate.distance(start.coordinate, geography);
	}
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
	}
	tracer(configuration &conf){
		config = conf;
		loadFrom(conf.trace_path.c_str());
		if(config.method == QTREE){
			part = new qtree_partitioner(mbr,config);
		}else if(config.method == FIX_GRID){
			part = new grid_partitioner(mbr,config);
		}
	};
	~tracer(){
		if(owned_trace){
			free(trace);
		}
		if(part){
			delete part;
		}
	}
	void process();
	void dumpTo(const char *path);
	void loadFrom(const char *path);
	void print_trace(double sample_rate);
	Point *get_trace(){
		return trace;
	}
};



#endif /* SRC_TRACING_TRACE_H_ */
