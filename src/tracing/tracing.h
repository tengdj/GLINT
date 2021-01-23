/*
 * tracing.h
 *
 *  Created on: Jan 19, 2021
 *      Author: teng
 */

#ifndef SRC_TRACING_TRACING_H_
#define SRC_TRACING_TRACING_H_

#include "../geometry/Map.h"
#include <map>

using namespace std;


class Grid{
	void rasterize(int num_grids);
public:
	double step = 0;
	int dimx = 0;
	int dimy = 0;
	box space;
	Grid(box &mbr, int num_grids){
		space = mbr;
		rasterize(num_grids);
	}
	int getgrid(Point *p);
	int getgrid(double x, double y);
	int get_grid_num(){
		return dimx*dimy;
	}
	Point get_random_point(int xoff=-1, int yoff=-1);
	double get_step(){
		return step/degree_per_kilometer_latitude;
	}
};

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
	double length = 0;
	double rate_sleep = 0;
	int max_sleep_time = 0;
	map<int,int> target_count;
	ZoneStats(int id){
		zoneid = id;
	}
	~ZoneStats(){
		target_count.clear();
	}
	double get_speed(){
		assert(length&&duration);
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
public:

    Map *map = NULL;
    configuration config;
	int counter = 0;

	// construct with some parameters
	trace_generator(configuration &c, Map *m){
		counter = c.num_objects;
		config = c;
		assert(config.num_threads>0);
		map = m;
		grid = new Grid(*m->getMBR(),config.num_grids);
		zones.resize(grid->dimx*grid->dimy+1);
		for(int i=0;i<zones.size();i++){
			zones[i] = new ZoneStats(i);
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
	int num_objects = 0;
	int duration = 0;
	Point *trace = NULL;
	bool owned_trace = false;
	// for query
	configuration config;
public:
	box mbr;
	tracer(configuration &conf, box &b, Point *t, int o, int d){
		trace = t;
		mbr = b;
		num_objects = o;
		duration = d;
		config = conf;
	}
	tracer(configuration &conf){
		loadFrom(conf.trace_path.c_str());
		config = conf;
		assert(config.duration<=duration);
		assert(config.num_objects<=num_objects);
	};
	~tracer(){
		if(owned_trace){
			free(trace);
		}
	}
	void process_qtree();
	void process_fixgrid();
	void dumpTo(const char *path);
	void loadFrom(const char *path);
};



#endif /* SRC_TRACING_TRACING_H_ */
