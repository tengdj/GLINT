/*
 * tracing.h
 *
 *  Created on: Jan 19, 2021
 *      Author: teng
 */

#ifndef SRC_TRACING_TRACE_H_
#define SRC_TRACING_TRACE_H_

#include "../geometry/Map.h"
#include "../util/query_context.h"
#include "partitioner.h"
#include "workbench.h"
#include <map>

#ifdef USE_GPU
#include "../cuda/mygpu.h"
#endif

using namespace std;


/*
 *
 * the statistics of the trips parsed from the taxi data
 * each zone
 *
 * */
class ZoneStats{
public:
	int zoneid = 0;
	long count = 0;
	long duration = 0;
	double length = 0.0;
	ZoneStats(int id){
		zoneid = id;
	}
	~ZoneStats(){
	}
};

class Event{
public:
	int timestamp;
	Point coordinate;
};

enum TripType{
	REST = 0,
	WALK = 1,
	DRIVE = 2
};

class Trip {
public:
	Event start;
	Event end;
	TripType type = REST;
	Trip(){};
	Trip(string str);
	void print_trip();
	int duration(){
		return end.timestamp-start.timestamp;
	}
	double speed(){
		return length()/duration();
	}
	double length(){
		return end.coordinate.distance(start.coordinate, true);
	}
	void resize(int max_duration);
};

class trace_generator{
	Grid *grid = NULL;
	vector<ZoneStats *> zones;
	ZoneStats *total = NULL;
public:

    Map *map = NULL;
    configuration *config = NULL;

	// construct with some parameters
	trace_generator(configuration *conf, Map *m){
		config = conf;
		assert(config->num_threads>0);
		map = m;
		grid = new Grid(*m->getMBR(),config->grid_width);
		zones.resize(grid->get_grid_num());
		for(int i=0;i<zones.size();i++){
			zones[i] = new ZoneStats(i);
		}
	}
	~trace_generator(){
		map = NULL;
		for(ZoneStats *z:zones){
			delete z;
		}
		zones.clear();
		if(grid){
			delete grid;
		}
		if(total){
			delete total;
		}
	}

	Point get_random_location();
	// generate the destination with the given source point
	Trip *next_trip(Trip *former=NULL);

	void analyze_trips(const char *path, int limit = 2147483647);
	Point *generate_trace();
	// generate a trace with given duration
	vector<Point *> get_trace(Map *mymap = NULL);

};


class tracer{
	// the statistic for the data set
	Point *trace = NULL;
	bool owned_trace = false;
	// for query
	configuration *config = NULL;
	partitioner *part = NULL;
	workbench *bench = NULL;
#ifdef USE_GPU
	gpu_info *gpu = NULL;
	workbench *d_bench = NULL;
#endif
public:
	box mbr;
	tracer(configuration *conf, box &b, Point *t){
		trace = t;
		mbr = b;
		config = conf;
		part = new partitioner(mbr,config);
#ifdef USE_GPU
		if(config->gpu){
			vector<gpu_info *> gpus = get_gpus();
			if(gpus.size()==0){
				log("not GPU is found, use CPU mode");
				config->gpu = false;
			}else{
				gpu = gpus[0];
				for(int i=1;i<gpus.size();i++){
					delete gpus[i];
				}
				gpus.clear();
			}
		}
#endif
	}
	tracer(configuration *conf){
		config = conf;
		loadFrom(config->trace_path.c_str());
		part = new partitioner(mbr,config);
#ifdef USE_GPU
		if(config->gpu){
			vector<gpu_info *> gpus = get_gpus();
			if(gpus.size()==0){
				log("not GPU is found, use CPU mode");
				config->gpu = false;
			}else{
				gpu = gpus[0];
				for(int i=1;i<gpus.size();i++){
					delete gpus[i];
				}
				gpus.clear();
			}
		}
#endif
	};
	~tracer(){
		if(owned_trace){
			free(trace);
		}
		if(part){
			delete part;
		}
		if(bench){
			delete bench;
		}
#ifdef USE_GPU
		if(gpu){
			delete gpu;
		}
#endif
	}
	void dumpTo(const char *path) {
		struct timeval start_time = get_cur_time();
		ofstream wf(path, ios::out|ios::binary|ios::trunc);
		wf.write((char *)&config->num_objects, sizeof(config->num_objects));
		wf.write((char *)&config->duration, sizeof(config->duration));
		wf.write((char *)&mbr, sizeof(mbr));
		size_t num_points = config->duration*config->num_objects;
		wf.write((char *)trace, sizeof(Point)*num_points);
		wf.close();
		logt("dumped to %s",start_time,path);
	}

	void loadFrom(const char *path) {

		int total_num_objects;
		int total_duration;
		struct timeval start_time = get_cur_time();
		ifstream in(path, ios::in | ios::binary);
		in.read((char *)&total_num_objects, sizeof(total_num_objects));
		in.read((char *)&total_duration, sizeof(total_duration));
		cout<<total_num_objects<<" "<<total_duration<<endl;
		in.read((char *)&mbr, sizeof(mbr));
		mbr.to_squre(true);
		assert(config->num_objects<=total_num_objects);
		assert(config->duration+config->start_time<=total_duration);

		in.seekg(config->start_time*total_num_objects*sizeof(Point), ios_base::cur);
		trace = (Point *)malloc(config->duration*config->num_objects*sizeof(Point));
		for(int i=0;i<config->duration;i++){
			in.read((char *)(trace+i*config->num_objects), config->num_objects*sizeof(Point));
			if(total_num_objects>config->num_objects){
				in.seekg((total_num_objects-config->num_objects)*sizeof(Point), ios_base::cur);
			}
		}
		in.close();
		logt("loaded %d objects last for %d seconds from %s",start_time, config->num_objects, config->duration, path);
		owned_trace = true;
	}

	void print(){
		double sample_rate = 1;
		if(config->num_objects>10000){
			sample_rate = 10000.0/config->num_objects;
		}
		print_points(trace,config->num_objects,sample_rate);
	}
	void print_trace(int oid){
		vector<Point *> points;
		for(int i=0;i<config->duration;i++){
			points.push_back(trace+oid+i*config->num_objects);
		}
		print_points(points);
		points.clear();
	}
	Point *get_trace(){
		return trace;
	}

	void process();

};



#endif /* SRC_TRACING_TRACE_H_ */
