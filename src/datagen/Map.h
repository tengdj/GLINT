/*
 * Map.h
 *
 *  Created on: Jan 11, 2021
 *      Author: teng
 */

#ifndef DATAGEN_MAP_H_
#define DATAGEN_MAP_H_

#include "geometry.h"
#include <queue>
#include <iostream>
#include <fstream>
#include <float.h>
#include "util/util.h"

using namespace std;
/*
 * represents a segment with some features
 *
 * */
class Street {
public:

	unsigned int id = 0;
	Point *start = NULL;
	Point *end = NULL;
	double length = -1.0;//Euclid distance of vector start->end
	vector<Street *> connected;
	vector<Street *> father_from_origin;

	~Street(){
		connected.clear();
		father_from_origin.clear();
	}
	void print(){
		printf("%d\t: ",id);
		printf("[[%f,%f],",start->x,start->y);
		printf("[%f,%f]]\t",end->x,end->y);
		printf("\tconnect: ");
		for(Street *s:connected) {
			printf("%d\t",s->id);
		}
		printf("\n");
	}


	//not distance in real world, but Euclid distance of the vector from start to end
	double getLength() {
		if(length<0) {
			length = start->distance(*end);
		}
		return length;
	}

	Street(unsigned int i, Point *s, Point *e) {
		assert(s&&e);
		start = s;
		end = e;
		id = i;
		start->connects.push_back(this);
		end->connects.push_back(this);
	}

	Point *close(Street *seg);

	//whether the target segment interact with this one
	//if so, put it in the connected map
	bool touch(Street *seg);



	/*
	 * commit a breadth-first search start from this
	 *
	 * */
	Street *breadthFirst(Street *target, int thread_id = 0);
};


/*
 *
 * the statistics of the trips parsed from the taxi data
 * each zone/hour
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


class Map {
	vector<Point *> nodes;
	vector<Street *> streets;
	vector<ZoneStats> zones;
	box *mbr = NULL;
	box *getMBR(){
		if(!mbr){
			mbr = new box();
			for(Point *p:nodes){
				mbr->update(*p);
			}
		}
		return mbr;
	}
	double step = 0;
	int dimx = 0;
	int dimy = 0;

public:
	~Map(){
		for(Street *s:streets){
			delete s;
		}
		for(Point *p:nodes){
			delete p;
		}
		zones.clear();
		if(mbr){
			delete mbr;
		}
	}

	void rasterize(int num_grids);
	int getgrid(Point *p);
	vector<Street *> getStreets(){
		return streets;
	}

	void set_thread_num(int thread_num){
		for(Street *s:streets){
			s->father_from_origin.resize(thread_num);
		}
	}
	void clear();
	void connect_segments();
	void dumpTo(const char *path);
	void loadFrom(const char *path);
	void loadFromCSV(const char *path);
	Street * nearest(Point *target);
	vector<Point *> navigate(Point *origin, Point *dest, double speed, int threadid = 0);
	vector<Point *> navigate(Trip *t, int threadid = 0){
		return navigate(&t->start.coordinate, &t->end.coordinate, t->duration(), threadid);
	}
	void print_region(box region);
	void analyze_trips(const char *path, int limit = 2147483647);
	vector<Point *> generate_trace(int thread_id=0, int start_time = 0, int end_time = 24*3600);
	Point *get_next(Point *original=NULL);
};

inline void print_linestring(vector<Point *> trajectory, double sample_rate=1.0){
	assert(sample_rate<=1&&sample_rate>0);
	printf("LINESTRING (");
	bool first = true;
	for(int i=0;i<trajectory.size();i++){

		if(tryluck(sample_rate)){
			if(!first){
				printf(",");
			}else{
				first = false;
			}
			printf("%f %f",trajectory[i]->x,trajectory[i]->y);
		}
	}

	printf(")\n");
}

vector<Trip *> load_trips(const char *path, int limit = 2147483647);



inline double distance_point_to_segment(Point *p, Street *s){
	return distance_point_to_segment(p->x,p->y,s->start->x,s->start->y,s->end->x,s->end->y);
}
#endif /* DATAGEN_MAP_H_ */
