/*
 * trace.cpp
 *
 *  Created on: Jan 22, 2021
 *      Author: teng
 */

#include "../tracing/trace.h"

#include "../geometry/Map.h"
#include "../util/config.h"
#include <vector>
#include <stdlib.h>

using namespace std;
#define PI 3.14159265

int main(int argc, char **argv){

	configuration config = get_parameters(argc, argv);
	config.print();
	tracer *tr = new tracer(config);

//	Map *m = new Map(config.map_path);
//	m->print_region(tr->mbr);
//	delete m;

	tr->process();

	delete tr;
	return 0;
}
