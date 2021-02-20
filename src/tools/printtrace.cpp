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
int main(int argc, char **argv){

	configuration config = get_parameters(argc, argv);
	config.print();
	tracer *tr = new tracer(&config);
	tr->print_trace();
	delete tr;
	return 0;
}
