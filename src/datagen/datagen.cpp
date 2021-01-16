/*
 * main.cpp
 *
 *  Created on: Jan 11, 2021
 *      Author: teng
 */


#include "Map.h"


int main(int argc, char **argv){

	struct timeval start = get_cur_time();

	Map *m = new Map();
//	m->loadFromCSV("/gisdata/chicago/streets.csv");
//	m->dumpTo("/gisdata/chicago/formated");
	m->loadFrom("/gisdata/chicago/formated");
	m->rasterize(500);
	m->analyze_trips("/gisdata/chicago/taxi.csv", 10000);
	logt("analyze trips",start);

	vector<Point *> trace = m->generate_trace(0,0,24*3600);
	print_linestring(trace,0.1);
	vector<Point *> trace2 = m->generate_trace(0,0,24*3600);
	print_linestring(trace2,0.1);
	logt("get trace",start);

//
//	int trip_count = 1000;
//	int duration = 0;
//	vector<Trip *> trips = load_trips("/gisdata/chicago/taxi.csv",trip_count);
//	for(int i=0;i<trip_count;i++){
//		m->navigate(trips[i]);
//		duration += trips[i]->duration();
//	}
//	trips[0]->print_trip();
//	vector<Point *> events = trips[0]->getTraces();
//	print_linestring(events);
//	print_linestring(trips[0]->trajectory);


	delete m;
	return 0;
}

