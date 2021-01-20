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
	double step = 0;
	int dimx = 0;
	int dimy = 0;
	vector<ZoneStats *> zones;
public:

    Map *map = NULL;
	int counter = 0;
	int duration = 0;
	int num_threads = 0;

	// construct with some parameters
	trace_generator(int ngrids, int cter, int dur, int thread, Map *m){
		counter = cter;
		duration = dur;
		num_threads = thread;
		if(num_threads<=0){
			num_threads = get_num_threads();
		}
		map = m;
		rasterize(ngrids);
	}
	~trace_generator(){
		map = NULL;
		for(ZoneStats *z:zones){
			delete z;
		}
		zones.clear();
	}
	void rasterize(int num_grids);
	int getgrid(Point *p);
	// generate the destination with the given source point
	Trip *next_trip(Trip *former=NULL);

	void analyze_trips(const char *path, int limit = 2147483647);
	double *generate_trace();
	// generate a trace with given duration
	vector<Point *> get_trace(Map *mymap);
};






#endif /* SRC_TRACING_TRACING_H_ */
