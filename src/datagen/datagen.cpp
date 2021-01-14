/*
 * main.cpp
 *
 *  Created on: Jan 11, 2021
 *      Author: teng
 */


#include "Map.h"


int main(int argc, char **argv){

	Map *m = new Map();
//	m->loadFromCSV("/gisdata/chicago/streets.csv");
//	m->dumpTo("/gisdata/chicago/formated");
	m->loadFrom("/gisdata/chicago/formated");
	m->rasterize(500);
	m->analyze_trips("/gisdata/chicago/taxi.csv", 100000);

//	trips[0]->navigate(m);
//	trips[0]->print_trip();
//	vector<Point *> events = trips[0]->getTraces();
//	print_linestring(events);
//	print_linestring(trips[0]->trajectory);


	delete m;
	return 0;
}

