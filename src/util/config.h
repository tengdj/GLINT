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

const static string process_method[] = {"QTree","GPU","fix grid"};
enum PROCESS_METHOD{
	QTREE = 0,
	GPU = 1,
	FIX_GRID = 2
};

class configuration{
public:
	int thread_id = 0;
	PROCESS_METHOD method = QTREE;
	int num_threads = get_num_threads();
	int duration = 1000;
	int num_objects = 1000;
	int num_grids = 1000;
	int num_trips = 100000;
	int max_objects_per_grid = 100;
	double reach_threshold = 5;
	string map_path = "/gisdata/chicago/streets";
	string taxi_path = "/gisdata/chicago/taxi.csv";
	string trace_path = "/gisdata/chicago/traces";
	configuration(){}
	void print(){
		printf("configuration:");
		printf("num threads:\t%d\n",num_threads);
		printf("num objects:\t%d\n",num_objects);
		printf("num grids:\t%d\n",num_grids);
		printf("num objects per grids:\t%d\n",max_objects_per_grid);
		printf("num trips:\t%d\n",num_trips);
		printf("duration:\t%d\n",duration);
		printf("reachable threshold:\t%f\n",reach_threshold);
		printf("map path:\t%s\n",map_path.c_str());
		printf("taxi path:\t%s\n",taxi_path.c_str());
		printf("trace path:\t%s\n",trace_path.c_str());
		printf("query method:\t%s\n",trace_path.c_str());

	}
};

inline configuration get_parameters(int argc, char **argv){
	configuration global_ctx;

	string query_method = "qtree";
	po::options_description desc("query usage");
	desc.add_options()
		("help,h", "produce help message")
		("threads,n", po::value<int>(&global_ctx.num_threads), "number of threads")
		("grids,g", po::value<int>(&global_ctx.num_grids), "number of grids")
		("num_objects_grid", po::value<int>(&global_ctx.max_objects_per_grid), "number of objects per grid")
		("trips,t", po::value<int>(&global_ctx.num_trips), "number of trips")
		("objects,o", po::value<int>(&global_ctx.num_objects), "number of objects")
		("duration,d", po::value<int>(&global_ctx.duration), "duration of the trace")
		("reachable_distance,r", po::value<double>(&global_ctx.reach_threshold), "reachable distance (in meters)")
		("map_path", po::value<string>(&global_ctx.map_path), "path to the map file")
		("taxi_path", po::value<string>(&global_ctx.taxi_path), "path to the taxi file")
		("trace_path", po::value<string>(&global_ctx.trace_path), "path to the trace file")
		("query_method,q", po::value<string>(&query_method), "query method (qtree|gpu|fix_grid)")
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
	}else if(query_method=="gpu"){
		global_ctx.method = GPU;
	}else{
		cerr <<"invalid query method "<<query_method<<endl;
		cerr << desc << "\n";
		exit(0);
	}

	return global_ctx;
}


#endif /* SRC_UTIL_CONTEXT_H_ */
