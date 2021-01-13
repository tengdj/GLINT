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

	vector<Trip *> trips = load_trips("/gisdata/chicago/taxi.csv", 1000000);
	int *hours = new int[24];
	for(int i=0;i<24;i++){
		hours[i] = 0;
	}
	for(Trip *tp:trips){
		hours[tp->start_time/3600]++;
	}
	for(int i=0;i<24;i++){
		cout<<hours[i]<<endl;
	}
	delete hours;

//	trips[0]->navigate(m);
//	trips[0]->print_trip();
//	vector<Point *> events = trips[0]->getTraces();
//	print_linestring(events);
//	print_linestring(trips[0]->trajectory);


	return 0;
}

