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

	context ctx;
	ctx.num_grids = 1000;
	ctx.duration = 1000;
	ctx.num_objects = 10000;
	ctx.num_trips = 100000;
	ctx.num_threads = get_num_threads();
	ctx.method = FIX_GRID;

	Map *m = new Map();
//	m->loadFromCSV("/gisdata/chicago/streets.csv");
//	m->dumpTo("/gisdata/chicago/formated");
	m->loadFrom("/gisdata/chicago/formated");
	struct timeval start = get_cur_time();
	trace_generator *gen = new trace_generator(ctx,m);
	gen->analyze_trips("/gisdata/chicago/taxi.csv", ctx.num_trips);
	logt("analyze trips",start);
	Point *traces = gen->generate_trace();
	logt("generate traces",start);

	tracer *tr = new tracer(ctx, *m->getMBR(), traces);
	tr->process();

	free(traces);
	delete gen;
	delete m;
	return 0;
}

