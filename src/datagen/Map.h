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

class Map {
	vector<Point *> nodes;
	vector<Street *> streets;
public:

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

#endif /* DATAGEN_MAP_H_ */
