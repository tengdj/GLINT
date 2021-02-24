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


typedef struct configuration{
	// shared parameters
	int thread_id = 0;
	uint num_threads = 1;
	uint duration = 1000;
	uint min_meet_time = 1;
	uint num_objects = 1000;
	string trace_path = "/gisdata/chicago/traces";

	// for query only
	uint start_time = 0;
	uint grid_capacity = 100;
	uint zone_capacity = 100;
	uint num_meeting_buckets = 100000;
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
}configuration;

inline void print_config(configuration &config){
	printf("configuration:");
	printf("num threads:\t%d\n",config.num_threads);
	printf("num objects:\t%d\n",config.num_objects);
	printf("grid capacity:\t%d\n",config.grid_capacity);
	printf("zone capacity:\t%d\n",config.zone_capacity);
	printf("start time:\t%d\n",config.start_time);
	printf("duration:\t%d\n",config.duration);
	printf("reach distance:\t%.0f m\n",config.reach_distance);
	printf("grid width:\t%.0f m\n",config.grid_width);
	printf("num meeting buckets:\t%d\n",config.num_meeting_buckets);

	printf("map path:\t%s\n",config.map_path.c_str());
	printf("taxi path:\t%s\n",config.taxi_path.c_str());
	printf("trace path:\t%s\n",config.trace_path.c_str());
	printf("use gpu:\t%s\n",config.gpu?"yes":"no");
	printf("analyze:\t%s\n",config.analyze?"yes":"no");

}

inline configuration get_parameters(int argc, char **argv){
	configuration config;
	config.num_threads = get_num_threads();

	string query_method = "qtree";
	po::options_description desc("query usage");
	desc.add_options()
		("help,h", "produce help message")
		("gpu,g", "use gpu for processing")
		("analyze,a", "analyze the processed data")
		("threads,n", po::value<uint>(&config.num_threads), "number of threads")
		("grid_capacity", po::value<uint>(&config.grid_capacity), "maximum number of objects per grid ")
		("zone_capacity", po::value<uint>(&config.zone_capacity), "maximum number of objects per zone buffer")
		("grid_width", po::value<double>(&config.grid_width), "the width of each grid (in meters)")
		("objects,o", po::value<uint>(&config.num_objects), "number of objects")
		("num_buckets,b", po::value<uint>(&config.num_meeting_buckets), "number of meeting buckets")

		("duration,d", po::value<uint>(&config.duration), "duration of the trace")
		("min_meet_time,t", po::value<uint>(&config.min_meet_time), "minimum meeting time")

		("start_time,s", po::value<uint>(&config.start_time), "the start time of the duration")

		("reachable_distance,r", po::value<double>(&config.reach_distance), "reachable distance (in meters)")
		("map_path", po::value<string>(&config.map_path), "path to the map file")
		("taxi_path", po::value<string>(&config.taxi_path), "path to the taxi file")
		("trace_path", po::value<string>(&config.trace_path), "path to the trace file")
		("distribution", po::value<double>(&config.distribution_rate), "percent of start points evenly distributed")

		;
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	if (vm.count("help")) {
		cout << desc << "\n";
		exit(0);
	}
	po::notify(vm);

	if(vm.count("analyze")){
		config.analyze = true;
	}
	if(vm.count("gpu")){
		config.gpu = true;
	}
	if(!vm.count("zone_capacity")||!vm.count("gpu")){
		config.zone_capacity = config.grid_capacity;
	}
	config.grid_width = max(config.grid_width, config.reach_distance);

	assert(config.walk_rate+config.drive_rate<=1);
	print_config(config);
	return config;
}


#endif /* SRC_UTIL_CONTEXT_H_ */
