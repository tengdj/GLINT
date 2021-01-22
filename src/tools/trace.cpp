/*
 * trace.cpp
 *
 *  Created on: Jan 22, 2021
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

	tracer *tr = new tracer(config.trace_path.c_str());
	if(config.method == QTREE){
		tr->process_qtree(config.num_grids);
	}else if(config.method == FIX_GRID){
		tr->process_fixgrid(config.num_grids);
	}

	delete tr;
	return 0;
}
