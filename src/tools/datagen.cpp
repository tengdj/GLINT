/*
 * main.cpp
 *
 *  Created on: Jan 11, 2021
 *      Author: teng
 */


#include "../geometry/Map.h"
#include "../tracing/tracing.h"
#include <vector>
#include <stdlib.h>

using namespace std;

int main(int argc, char **argv){

	int grid_num = 100000;
	int duration = 1000;
	int num_objects = 1000;
	int num_trips = 10000;

	Map *m = new Map();
//	m->loadFromCSV("/gisdata/chicago/streets.csv");
//	m->dumpTo("/gisdata/chicago/formated");
	m->loadFrom("/gisdata/chicago/formated");
	struct timeval start = get_cur_time();
	trace_generator *gen = new trace_generator(grid_num,num_objects,duration, get_num_threads(),m);
	gen->analyze_trips("/gisdata/chicago/taxi.csv", num_trips);
	logt("analyze trips",start);
	double *traces = gen->generate_trace();
	delete gen;
	logt("generate traces",start);

	// test contact tracing
	vector<vector<Point *>> grids;
	grids.resize(grid_num);
	for(int i=0;i<duration*num_objects;i++){
		Point *p = new Point(traces[i*2],traces[i*2+1]);
		int g = i%grid_num;//m->getgrid(p);
		grids[g].push_back(p);
	}
	logt("get grid",start);
	int index = 0;
	double mindist = DBL_MAX;
	int count = 0;
	for(vector<Point *> &ps:grids){
		int len = ps.size();
		index++;
		if(len>0){
			//cout<<index<<" "<<len<<endl;
		}else{
			continue;
		}
		for(int i=0;i<len-1;i++){
			for(int j=i+1;j<ps.size();j++){
				double dist = ps[i]->distance(*ps[j], true);
				count++;
				if(dist<mindist){
					mindist = dist;
				}
			}
		}
	}

	logt("contact %d",start,count);

	delete []traces;
//	vector<Point *> trace2 = m->generate_trace(0,0,24*3600);
//	print_linestring(trace2,0.1);
//	logt("get trace",start);

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

