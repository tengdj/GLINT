/*
 * tracing.h
 *
 *  Created on: Jan 19, 2021
 *      Author: teng
 */

#ifndef SRC_TRACING_TRACING_H_
#define SRC_TRACING_TRACING_H_

#include "../geometry/Map.h"




/*
 *
 * the statistics of the trips parsed from the taxi data
 * each zone
 *
 * */
class ZoneStats{
	int zoneid;
public:
	long count = 0;
	long duration = 0;
	double length = 0;
	double rate_sleep = 0;
	vector<double> rate_target;
	~ZoneStats(){
		rate_target.clear();
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
	Trip(string str);
	void print_trip();
	int duration(){
		return end.timestamp-start.timestamp;
	}
};

class trace_generator{
	double step = 0;
	int dimx = 0;
	int dimy = 0;
	vector<ZoneStats> zones;
	void rasterize(int num_grids);
	int getgrid(Point *p);
	// generate the destination with the given source point
	Point *get_next(Point *original=NULL);
public:

    Map *map = NULL;
	int counter = 0;
	int duration = 0;
	int num_threads = 0;

	// construct with some parameters
	trace_generator(int num_grids, int cter, int dur, int thread, Map *m){
		counter = cter;
		duration = dur;
		num_threads = thread;
		if(num_threads<=0){
			num_threads = get_num_threads();
		}
		map = m;
		rasterize(num_grids);

	}
	void analyze_trips(const char *path, int limit = 2147483647);
	double *generate_trace();
	// generate a trace with given duration
	vector<Point *> get_trace(Map *mymap);
};






#endif /* SRC_TRACING_TRACING_H_ */
