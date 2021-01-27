/*
 * trace.cpp
 *
 *  Created on: Jan 22, 2021
 *      Author: teng
 */

#include "../geometry/Map.h"
#include "../tracing/tracing.h"
#include "../util/config.h"
#include <vector>
#include <stdlib.h>

using namespace std;
#define PI 3.14159265

int main(int argc, char **argv){

	configuration config = get_parameters(argc, argv);
	config.print();
	tracer *tr = new tracer(config);
	if(config.method == QTREE){
		tr->process_qtree();
	}else if(config.method == FIX_GRID){
		tr->process_fixgrid();
	}
	delete tr;
	return 0;
}
