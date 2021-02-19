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


class configuration{
public:
	// shared parameters
	int thread_id = 0;
	int num_threads = get_num_threads();
	int duration = 1000;
	uint num_objects = 1000;
	string trace_path = "/gisdata/chicago/traces";


	// for query only
	int start_time = 0;
	uint num_objects_per_round = 1000000;
	int num_trips = 100000;
	int grid_capacity = 100;
	int zone_capacity = 100;
	double reach_distance = 5;
	double x_buffer = 0;
	double y_buffer = 0;
	bool gpu = false;
	bool analyze = false;

	// for generator only
	double grid_width = 5;
	string map_path = "/gisdata/chicago/streets";
	string taxi_path = "/gisdata/chicago/taxi.csv";
	// how many percent of the initial points are evenly distributed
	double distribution_rate = 0.3;
	double walk_rate = 0.4;
	double walk_speed = 1.0;
	double drive_rate = 0.2;
	double drive_speed = 15.0;


	configuration(){}
	void print(){
		printf("configuration:");
		printf("num threads:\t%d\n",num_threads);
		printf("num objects:\t%d\n",num_objects);
		printf("num objects per grid:\t%d\n",grid_capacity);
		printf("num objects per zone:\t%d\n",zone_capacity);
		printf("num trips:\t%d\n",num_trips);
		printf("start time:\t%d\n",start_time);
		printf("duration:\t%d\n",duration);
		printf("reach threshold:\t%f m\n",reach_distance);
		printf("grid width:\t%f m\n",grid_width);

		printf("map path:\t%s\n",map_path.c_str());
		printf("taxi path:\t%s\n",taxi_path.c_str());
		printf("trace path:\t%s\n",trace_path.c_str());
		printf("use gpu:\t%s\n",gpu?"yes":"no");
		printf("analyze:\t%s\n",analyze?"yes":"no");

	}
};

inline configuration get_parameters(int argc, char **argv){
	configuration global_ctx;

	string query_method = "qtree";
	po::options_description desc("query usage");
	desc.add_options()
		("help,h", "produce help message")
		("gpu,g", "use gpu for processing")
		("analyze,a", "analyze the processed data")
		("threads,n", po::value<int>(&global_ctx.num_threads), "number of threads")
		("grid_capacity", po::value<int>(&global_ctx.grid_capacity), "maximum number of objects per grid ")
		("zone_capacity", po::value<int>(&global_ctx.zone_capacity), "maximum number of objects per zone buffer")
		("grid_width", po::value<double>(&global_ctx.grid_width), "the width of each grid (in meters)")
		("trips,t", po::value<int>(&global_ctx.num_trips), "number of trips")
		("objects,o", po::value<uint>(&global_ctx.num_objects), "number of objects")
		("duration,d", po::value<int>(&global_ctx.duration), "duration of the trace")
		("start_time,s", po::value<int>(&global_ctx.start_time), "the start time of the duration")

		("reachable_distance,r", po::value<double>(&global_ctx.reach_distance), "reachable distance (in meters)")
		("map_path", po::value<string>(&global_ctx.map_path), "path to the map file")
		("taxi_path", po::value<string>(&global_ctx.taxi_path), "path to the taxi file")
		("trace_path", po::value<string>(&global_ctx.trace_path), "path to the trace file")
		("distribution", po::value<double>(&global_ctx.distribution_rate), "percent of start points evenly distributed")

		;
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	if (vm.count("help")) {
		cout << desc << "\n";
		exit(0);
	}
	po::notify(vm);

	if(vm.count("analyze")){
		global_ctx.analyze = true;
	}
	if(vm.count("gpu")){
		global_ctx.gpu = true;
	}
	if(!vm.count("zone_capacity")){
		global_ctx.zone_capacity = global_ctx.grid_capacity;
	}

	global_ctx.grid_width = max(global_ctx.grid_width, global_ctx.reach_distance);

	assert(global_ctx.walk_rate+global_ctx.drive_rate<=1);

	return global_ctx;
}


#endif /* SRC_UTIL_CONTEXT_H_ */
