/*
 * main.cpp
 *
 *  Created on: Jan 11, 2021
 *      Author: teng
 */


#include "../geometry/Map.h"
#include "../tracing/tracing.h"
#include "../util/context.h"
#include <vector>
#include <stdlib.h>

using namespace std;

int main(int argc, char **argv){

	configuration config = get_parameters(argc, argv);

	Map *m = new Map(config.map_path+".csv");
	trace_generator *gen = new trace_generator(config,m);
	gen->analyze_trips(config.taxi_path.c_str(), config.num_trips);
	Point *traces = gen->generate_trace();
	tracer *t = new tracer(config,*m->getMBR(),traces);
	t->dumpTo(config.trace_path.c_str());

	free(traces);
	delete gen;
	delete m;
	delete t;
	return 0;
}

