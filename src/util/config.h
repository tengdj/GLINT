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
	uint num_threads = 1;
	uint duration = 1000;
	uint num_objects = 1000;
	string trace_path = "/gisdata/chicago/traces";

	// for query only
	uint start_time = 0;
	uint grid_capacity = 100;
	uint zone_capacity = 100;
	uint num_meeting_buckets = 100000;
	uint bucket_size = 20;
	uint num_meeting_buckets_overflow = 1000;
	uint refine_size = 3;
	bool dynamic_schema = false;
	bool phased_lookup = false;
	bool unroll = false;
	uint schema_update_delay = 1; //
	uint min_meet_time = 10;
	double reach_distance = 2;
	double x_buffer = 0;
	double y_buffer = 0;
	bool gpu = false;
	uint specific_gpu = 0;
	bool analyze_reach = false;
	bool analyze_grid = false;
	bool analyze_meeting = false;
	bool profile = false;

	void print(){
		printf("configuration:\n");
		printf("num threads:\t%d\n",num_threads);
		printf("num objects:\t%d\n",num_objects);
		printf("grid capacity:\t%d\n",grid_capacity);
		printf("zone capacity:\t%d\n",zone_capacity);
		printf("start time:\t%d\n",start_time);
		printf("duration:\t%d\n",duration);
		printf("reach distance:\t%.0f m\n",reach_distance);
		printf("minimum meeting time:\t%d\n",min_meet_time);
		printf("bucket size:\t%d\n",bucket_size);
		printf("num buckets:\t%d\n",num_meeting_buckets);

		printf("trace path:\t%s\n",trace_path.c_str());
		printf("use gpu:\t%s\n",gpu?"yes":"no");
		if(gpu){
			printf("which gpu:\t%d\n",specific_gpu);
		}
		printf("dynamic schema:\t%s\n",dynamic_schema?"yes":"no");
		printf("schema update gap:\t%d\n",schema_update_delay);

		printf("analyze reach:\t%s\n",analyze_reach?"yes":"no");
		printf("analyze grid:\t%s\n",analyze_grid?"yes":"no");
		printf("analyze meeting:\t%s\n",analyze_meeting?"yes":"no");

	}
};


inline configuration get_parameters(int argc, char **argv){
	configuration config;
	config.num_threads = get_num_threads();

	po::options_description desc("query usage");
	desc.add_options()
		("help,h", "produce help message")
		("gpu,g", "use gpu for processing")
		("profile,p", "profile the memory usage")
		("phased_filter", "enable phased filter")
		("unroll,u", "unroll the refinement")

		("analyze_reach", "analyze the reaches statistics")
		("analyze_meeting", "analyze the meeting bucket statistics")
		("analyze_grid", "analyze the grid statistics")
		("threads,n", po::value<uint>(&config.num_threads), "number of threads")
		("specific_gpu", po::value<uint>(&config.specific_gpu), "use which gpu")
		("grid_capacity", po::value<uint>(&config.grid_capacity), "maximum number of objects per grid ")
		("zone_capacity", po::value<uint>(&config.zone_capacity), "maximum number of objects per zone buffer")
		("refine_size", po::value<uint>(&config.refine_size), "number of refine list entries per object")
		("objects,o", po::value<uint>(&config.num_objects), "number of objects")
		("num_buckets,b", po::value<uint>(&config.num_meeting_buckets), "number of meeting buckets")
		("bucket_size", po::value<uint>(&config.bucket_size), "the size of each bucket")

		("duration,d", po::value<uint>(&config.duration), "duration of the trace")
		("min_meet_time,m", po::value<uint>(&config.min_meet_time), "minimum meeting time")

		("start_time,s", po::value<uint>(&config.start_time), "the start time of the duration")

		("reachable_distance,r", po::value<double>(&config.reach_distance), "reachable distance (in meters)")
		("trace_path,t", po::value<string>(&config.trace_path), "path to the trace file")
		("dynamic_schema", "the schema is dynamically updated")

		;
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	if (vm.count("help")) {
		cout << desc << "\n";
		exit(0);
	}
	po::notify(vm);

	if(vm.count("gpu")){
		config.gpu = true;
	}
	if(vm.count("analyze_reach")){
		config.analyze_reach = true;
	}
	if(vm.count("analyze_meeting")){
		config.analyze_meeting = true;
	}
	if(vm.count("analyze_grid")){
		config.analyze_grid = true;
	}
	if(vm.count("profile")){
		config.profile = true;
	}
	if(vm.count("dynamic_schema")){
		config.dynamic_schema = true;
	}

	if(!vm.count("zone_capacity")||!vm.count("gpu")){
		config.zone_capacity = config.grid_capacity;
	}
	if(!vm.count("num_buckets")){
		config.num_meeting_buckets = config.num_objects;
	}
	if(vm.count("phased_filter")){
		config.phased_lookup = true;
	}
	if(vm.count("unroll")){
		config.unroll = true;
	}else{
		config.zone_capacity = config.grid_capacity;
	}

	if(!vm.count("bucket_size")){
		config.bucket_size = 5.0*config.num_objects/config.num_meeting_buckets;
	}

	config.print();
	return config;
}


class generator_configuration:public configuration{
public:
	// how many percent of the initial points are evenly distributed
	double walk_rate = 0.4;
	double walk_speed = 1.0;
	double drive_rate = 0.2;
	double drive_speed = 15.0;
	int max_rest_time = 600;

	string map_path = "/gisdata/chicago/streets";
	string meta_path = "/gisdata/chicago/tweet.dat";

	void print(){
		printf("generator configuration:\n");
		printf("num threads:\t%d\n",num_threads);
		printf("num objects:\t%d\n",num_objects);
		printf("duration:\t%d\n",duration);
		printf("walk rate:\t%.2f\n",walk_rate);
		printf("walk speed:\t%.2f\n",walk_speed);
		printf("drive rate:\t%.2f\n",drive_rate);
		printf("drive speed:\t%.2f\n",drive_speed);

		printf("map path:\t%s\n",map_path.c_str());
		printf("metadata path:\t%s\n",meta_path.c_str());
		printf("trace path:\t%s\n",trace_path.c_str());
	}
};


inline generator_configuration get_generator_parameters(int argc, char **argv){
	generator_configuration config;
	config.num_threads = get_num_threads();

	po::options_description desc("generator usage");
	desc.add_options()
		("help,h", "produce help message")
		("threads,n", po::value<uint>(&config.num_threads), "number of threads")
		("objects,o", po::value<uint>(&config.num_objects), "number of objects")
		("duration,d", po::value<uint>(&config.duration), "duration of the trace")
		("map_path", po::value<string>(&config.map_path), "path to the map file")
		("trace_path", po::value<string>(&config.trace_path), "path to the trace file")
		("meta_path", po::value<string>(&config.meta_path), "path to the metadata file")
		("walk_rate", po::value<double>(&config.walk_rate), "percent of walk")
		("walk_speed", po::value<double>(&config.walk_speed), "the speed of walk (meters/second)")
		("drive_rate", po::value<double>(&config.drive_rate), "percent of drive")
		("drive_speed", po::value<double>(&config.drive_speed), "the speed of drive (meters/second)")
		;
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	if (vm.count("help")) {
		cout << desc << "\n";
		exit(0);
	}
	po::notify(vm);

	assert(config.walk_rate+config.drive_rate<=1);
	config.print();
	return config;
}


#endif /* SRC_UTIL_CONTEXT_H_ */
