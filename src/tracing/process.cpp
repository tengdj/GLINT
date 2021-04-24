/*
 * process.cpp
 *
 *  Created on: Feb 11, 2021
 *      Author: teng
 */

#include "trace.h"
#include "../index/QTree.h"



/*
 * functions for tracer
 *
 * */



tracer::tracer(configuration *conf, box &b, Point *t){
	trace = t;
	mbr = b;
	config = conf;
	part = new partitioner(mbr,config);
#ifdef USE_GPU
	if(config->gpu){
		vector<gpu_info *> gpus = get_gpus();
		if(gpus.size()==0){
			log("no GPU is found, use CPU mode");
			config->gpu = false;
		}else{
			assert(config->specific_gpu<gpus.size());
			gpu = gpus[config->specific_gpu];
			gpu->print();
			for(int i=0;i<gpus.size();i++){
				if(i!=config->specific_gpu){
					delete gpus[i];
				}
			}
			gpus.clear();
		}
	}
#endif
}
tracer::tracer(configuration *conf){
	config = conf;
	loadMeta(config->trace_path.c_str());
	part = new partitioner(mbr,config);
#ifdef USE_GPU
	if(config->gpu){
		vector<gpu_info *> gpus = get_gpus();
		if(gpus.size()==0){
			log("not GPU is found, use CPU mode");
			config->gpu = false;
		}else{
			assert(config->specific_gpu<gpus.size());
			gpu = gpus[config->specific_gpu];
			gpu->print();
			for(int i=0;i<gpus.size();i++){
				if(i!=config->specific_gpu){
					delete gpus[i];
				}
			}
			gpus.clear();
		}
	}
#endif
};
tracer::~tracer(){
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
void tracer::dumpTo(const char *path) {
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

void tracer::loadMeta(const char *path) {

	uint true_num_objects;
	uint true_duration;
	ifstream in(path, ios::in | ios::binary);
	if(!in.is_open()){
		log("%s cannot be opened",path);
		exit(0);
	}
	in.read((char *)&true_num_objects, sizeof(true_num_objects));
	in.read((char *)&true_duration, sizeof(true_duration));
	log("%d objects last for %d seconds in file",true_num_objects,true_duration);
	in.read((char *)&mbr, sizeof(mbr));
	mbr.to_squre(true);
	mbr.print();
	assert((size_t)config->num_objects*(config->start_time+config->duration)<=(size_t)true_num_objects*true_duration);
	//assert(config->num_objects<=true_num_objects);
	assert(config->start_time+config->duration<=true_duration);
	in.close();
}

void tracer::loadData(const char *path, int st, int duration) {

	log("loading locations from %d to %d",st,st+duration);
	assert(duration<=100);
	uint true_num_objects;
	uint true_duration;
	box mbr;
	struct timeval start_time = get_cur_time();
	ifstream in(path, ios::in | ios::binary);
	if(!in.is_open()){
		log("%s cannot be opened",path);
		exit(0);
	}
	in.read((char *)&true_num_objects, sizeof(true_num_objects));
	in.read((char *)&true_duration, sizeof(true_duration));
	in.read((char *)&mbr, sizeof(mbr));

	assert((size_t)config->num_objects*(st+duration)<=(size_t)true_num_objects*true_duration);
	//assert(config->num_objects<=true_num_objects);
	assert(st+duration<=true_duration);

	in.seekg(st*true_num_objects*sizeof(Point), ios_base::cur);
	if(!trace){
		trace = (Point *)malloc(min((uint)100,config->duration)*config->num_objects*sizeof(Point));
	}

	uint loaded = 0;
	while(loaded<config->num_objects){
		uint cur_num_objects = min(true_num_objects,config->num_objects-loaded);
		for(int i=0;i<duration;i++){
			in.read((char *)(trace+i*config->num_objects+loaded), cur_num_objects*sizeof(Point));
			if(true_num_objects>cur_num_objects){
				in.seekg((true_num_objects-cur_num_objects)*sizeof(Point), ios_base::cur);
			}
		}
		loaded += cur_num_objects;
	}

	in.close();
	logt("loaded %d objects last for %d seconds start from %d time from %s",start_time, config->num_objects, duration, st, path);
	owned_trace = true;
}

void tracer::print(){
	print_points(trace,config->num_objects,min(config->num_objects,(uint)10000));
}
void tracer::print_trace(int oid){
	vector<Point *> points;
	for(int i=0;i<config->duration;i++){
		points.push_back(trace+i*config->num_objects+oid);
	}
	print_points(points);
	points.clear();
}
void tracer::print_traces(){
	vector<Point *> points;
	for(int oid=0;oid<config->num_objects;oid++){
		for(int i=0;i<config->duration;i++){
			points.push_back(trace+i*config->num_objects+oid);
		}
	}
	print_points(points, 10000);
	points.clear();
}

#ifdef USE_GPU
workbench *create_device_bench(workbench *bench, gpu_info *gpu);
void process_with_gpu(workbench *bench,workbench *d_bench, gpu_info *gpu);
#endif

void tracer::process(){
	struct timeval start = get_cur_time();

	for(int st=config->start_time;st<config->start_time+config->duration;st+=100){
		int cur_duration = min((config->start_time+config->duration-st),(uint)100);
		this->loadData(config->trace_path.c_str(),st,cur_duration);
		if(!bench){
			bench = part->build_schema(trace, config->num_objects);
#ifdef USE_GPU
			if(config->gpu){
				d_bench = create_device_bench(bench, gpu);
			}
#endif
		}
		for(int t=0;t<cur_duration;t++){
			log("");
			bench->reset();
			bench->points = trace+t*config->num_objects;
			bench->cur_time = t;
			// process the coordinate in this time point
			if(!config->gpu){
				struct timeval ct = get_cur_time();
				bench->filter();
				bench->pro.filter_time += get_time_elapsed(ct,true);
				bench->reachability();
				bench->pro.refine_time += get_time_elapsed(ct,true);
				bench->update_meetings();
				bench->pro.meeting_update_time += get_time_elapsed(ct,true);
			}else{
	#ifdef USE_GPU
				process_with_gpu(bench,d_bench,gpu);
	#endif
			}
			if(bench->config->analyze_meeting&&bench->meeting_counter>0&&t==config->duration-1){
				int luck = get_rand_number(min(bench->meeting_counter, bench->meeting_capacity));
				print_trace(bench->meetings[luck].pid1);
				print_trace(bench->meetings[luck].pid2);
			}
			if(config->analyze_grid||config->profile){
				bench->analyze_grids();
			}
			if(config->analyze_reach){
				bench->analyze_reaches();
			}
			if(config->analyze_meeting||config->profile){
				bench->analyze_meeting_buckets();
			}
			if(config->dynamic_schema&&!config->gpu){
				struct timeval ct = get_cur_time();
				bench->update_schema();
				bench->pro.index_update_time += get_time_elapsed(ct,true);
			}
			logt("round %d",start,st+t+1);
			bench->current_bucket = !bench->current_bucket;
			bench->pro.rounds++;
			if(bench->pro.max_filter_size<bench->grid_check_counter){
				bench->pro.max_filter_size = bench->grid_check_counter;
			}
		}

	}



	bench->print_profile();
}
