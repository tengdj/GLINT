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

	Street *father_from_origin = NULL;
	double dist_from_origin = 0;

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
		start = s;
		end = e;
		id = i;
	}
	Street() {

	}

	Point *close(Street *seg);

	//whether the target segment interact with this one
	//if so, put it in the connected map
	bool touch(Street *seg);



	/*
	 * commit a breadth-first search start from this
	 *
	 * */
	Street *breadthFirst(long target_id);
};


/*
 *
 * the statistics of the trips parsed from the taxi data
 * each zone/hour
 *
 * */
class ZoneStats{
public:
	long count = 0;
	int zoneid;
	int timestamp;
	double speed;
	double rate_sleep = 0;
	vector<double> rate_target;
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

	void rasterize(int num_grids);
	int getgrid(Point *p);
	vector<Street *> getStreets(){
		return streets;
	}

	void clear();
	void connect_segments();
	void dumpTo(const char *path);
	void loadFrom(const char *path);
	void loadFromCSV(const char *path);
	Street * nearest(Point *target);
	vector<Point *> navigate(Point *origin, Point *dest);
	void print_region(box region);
	void analyze_trips(const char *path, int limit = 2147483647);
};

class Event{
public:
	int timestamp;
	Point coordinate;
};

class Trip {
public:
	//note that the time in the Chicago taxi dataset is messed up
	//the end_time and start_time is rounded to the nearest 15 minute like 0, 15, 45.
	//the duration_time is rounded to the nearest 10 seconds
	//thus for most cases end_time-start_time != duration_time
	int start_time;
	int end_time;
	Point start_location;
	Point end_location;
	vector<Point *> trajectory;
	Trip(string str);
	/*
	 * simulate the trajectory of the trip.
	 * with the given streets the trip has covered, generate a list of points
	 * that the taxi may appear at a given time
	 *
	 * */
	vector<Point *> getTraces();
	void navigate(Map *m);
	void print_trip();

};

inline void print_linestring(vector<Point *> trajectory){
	printf("LINESTRING (");
	for(int i=0;i<trajectory.size();i++){
		if(i>0){
			printf(",");
		}
		printf("%f %f",trajectory[i]->x,trajectory[i]->y);
	}

	printf(")\n");
}

vector<Trip *> load_trips(const char *path, int limit = 2147483647);



inline double distance_point_to_segment(Point *p, Street *s){
	return distance_point_to_segment(p->x,p->y,s->start->x,s->start->y,s->end->x,s->end->y);
}
#endif /* DATAGEN_MAP_H_ */
