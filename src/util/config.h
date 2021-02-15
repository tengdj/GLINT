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
	int num_trips = 100000;
	int grid_capacity = 100;
	double reach_distance = 5;
	// for grid partitioning
	double grid_width = 5;
	bool gpu = false;

	string map_path = "/gisdata/chicago/streets";
	string taxi_path = "/gisdata/chicago/taxi.csv";
	string trace_path = "/gisdata/chicago/traces";
	configuration(){}
	void print(){
		printf("configuration:");
		printf("num threads:\t%d\n",num_threads);
		printf("num objects:\t%d\n",num_objects);
		printf("num objects per grids:\t%d\n",grid_capacity);
		printf("num trips:\t%d\n",num_trips);
		printf("duration:\t%d\n",duration);
		printf("reach threshold:\t%f m\n",reach_distance);
		printf("grid width:\t%f m\n",grid_width);

		printf("map path:\t%s\n",map_path.c_str());
		printf("taxi path:\t%s\n",taxi_path.c_str());
		printf("trace path:\t%s\n",trace_path.c_str());
		printf("query method:\t%s\n",process_method[method].c_str());
	}
};

inline configuration get_parameters(int argc, char **argv){
	configuration global_ctx;

	string query_method = "qtree";
	po::options_description desc("query usage");
	desc.add_options()
		("help,h", "produce help message")
		("gpu,g", "use gpu for processing")
		("threads,n", po::value<int>(&global_ctx.num_threads), "number of threads")
		("grid_capacity", po::value<int>(&global_ctx.grid_capacity), "maximum number of objects per grid zone buffer")
		("grid_width", po::value<double>(&global_ctx.grid_width), "the width of each grid (in meters)")
		("trips,t", po::value<int>(&global_ctx.num_trips), "number of trips")
		("objects,o", po::value<int>(&global_ctx.num_objects), "number of objects")
		("duration,d", po::value<int>(&global_ctx.duration), "duration of the trace")
		("reachable_distance,r", po::value<double>(&global_ctx.reach_distance), "reachable distance (in meters)")
		("map_path", po::value<string>(&global_ctx.map_path), "path to the map file")
		("taxi_path", po::value<string>(&global_ctx.taxi_path), "path to the taxi file")
		("trace_path", po::value<string>(&global_ctx.trace_path), "path to the trace file")
		("query_method,q", po::value<string>(&query_method), "query method (qtree|gpu|grid)")
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
	}else if(query_method=="grid"){
		global_ctx.method = FIX_GRID;
	}else if(query_method=="gpu"){
		global_ctx.method = GPU;
	}else{
		cerr <<"invalid query method "<<query_method<<endl;
		cerr << desc << "\n";
		exit(0);
	}
	if(vm.count("gpu")){
		global_ctx.gpu = true;
	}

	global_ctx.grid_width = max(global_ctx.grid_width, global_ctx.reach_distance/sqrt(2));

	return global_ctx;
}


#endif /* SRC_UTIL_CONTEXT_H_ */
