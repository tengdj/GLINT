/*
 * context.h
 *
 *  Created on: Jan 16, 2021
 *      Author: teng
 */

#ifndef SRC_UTIL_CONTEXT_H_
#define SRC_UTIL_CONTEXT_H_

#include <boost/program_options.hpp>

#include "util.h"
namespace po = boost::program_options;

enum PROCESS_METHOD{
	QTREE = 0,
	GPU = 1,
	FIX_GRID = 2
};

class configuration{
public:
	int thread_id = 0;
	PROCESS_METHOD method = QTREE;
	int duration = 1000;
	int num_threads = get_num_threads();
	int num_objects = 1000;
	int num_grids = 1000;
	int num_trips = 100000;
	double reach_threshold = 0.1;
	void *target[3] = {NULL,NULL,NULL};
	string map_path = "/gisdata/chicago/streets";
	string taxi_path = "/gisdata/chicago/taxi.csv";
	string trace_path = "/gisdata/chicago/traces";
	configuration(){}
};

inline configuration get_parameters(int argc, char **argv){
	configuration global_ctx;

	string query_method = "qtree";
	po::options_description desc("query usage");
	desc.add_options()
		("help,h", "produce help message")
		("threads,n", po::value<int>(&global_ctx.num_threads), "number of threads")
		("grids,g", po::value<int>(&global_ctx.num_grids), "number of grids")
		("trips,t", po::value<int>(&global_ctx.num_threads), "number of trips")
		("objects,o", po::value<int>(&global_ctx.num_objects), "number of objects")
		("duration,d", po::value<int>(&global_ctx.duration), "duration of the trace")
		("map_path", po::value<string>(&global_ctx.map_path), "path to the map file")
		("taxi_path", po::value<string>(&global_ctx.taxi_path), "path to the taxi file")
		("trace_path", po::value<string>(&global_ctx.trace_path), "path to the trace file")
		("query_method,q", po::value<string>(&query_method), "query method (qtree|fix_grid)")
		;
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	if (vm.count("help")) {
		cout << desc << "\n";
		exit(0);
	}
	po::notify(vm);
	if(query_method=="qtree"){
		global_ctx.method = QTREE;
	}else if(query_method=="fix_grid"){
		global_ctx.method = FIX_GRID;
	}else{
		cerr <<"invalid query method "<<query_method<<endl;
		cerr << desc << "\n";
		exit(0);
	}

	return global_ctx;
}


#endif /* SRC_UTIL_CONTEXT_H_ */
