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

void *process_grid_unit(void *arg){
	query_context *ctx = (query_context *)arg;
	partition_info *pinfo = (partition_info *)ctx->target[0];
	Point *points = pinfo->points;
	uint *result = (uint *)ctx->target[1];
	uint *grid_check = pinfo->grid_checkings;
	size_t checked = 0;
	size_t reached = 0;
	while(true){
		// pick one batch of point-grid pair for processing
		size_t start = 0;
		size_t end = 0;
		if(!ctx->next_batch(start,end)){
			break;
		}
		for(uint pairid=start;pairid<end;pairid++){
			uint pid = grid_check[2*pairid];
			uint zid = grid_check[2*pairid+1];
			//log("%d\t%d",pid,gid);
			uint *cur_pids = pinfo->get_zone(zid);
			result[pid] = 0;
			//vector<Point *> pts;
			Point *p1 = points + pid;
			for(uint i=0;i<pinfo->get_zone_size(zid);i++){
				//pts.push_back(points + cur_pids[i]);
				Point *p2 = points + cur_pids[i];
				if(p1!=p2){
					//log("%f",dist);
					bool indist = p1->distance(*p2, true)<=ctx->config.reach_distance;
					result[pid] += indist;
					result[cur_pids[i]] += indist;
					reached += 2*indist;
					checked++;
				}
			}
		}
	}
	lock();
	ctx->checked += checked;
	ctx->found += reached;
	unlock();
	return NULL;
}

void process_with_cpu(query_context &tctx){
	struct timeval start = get_cur_time();
	pthread_t threads[tctx.config.num_threads];
	tctx.clear();

	for(int i=0;i<tctx.config.num_threads;i++){
		pthread_create(&threads[i], NULL, process_grid_unit, (void *)&tctx);
	}
	for(int i = 0; i < tctx.config.num_threads; i++ ){
		void *status;
		pthread_join(threads[i], &status);
	}
	logt("compute",start);
}

#ifdef USE_GPU
void process_with_gpu(query_context &ctx);
#endif

void tracer::process(){

	struct timeval start = get_cur_time();
	// test contact tracing
	size_t checked = 0;
	size_t reached = 0;
	query_context qctx;
	qctx.config = config;
	for(int t=0;t<config.duration;t++){

		Point *cur_trace = trace+t*config.num_objects;
		partition_info *pinfo = part->partition(cur_trace, config.num_objects);
		qctx.target[0] = (void *)pinfo;
		qctx.num_objects = pinfo->num_grid_checkings;
		qctx.target[1] = result;
		// process the objects in the packed partitions
		if(!config.gpu){
			process_with_cpu(qctx);
		}else{
#ifdef USE_GPU
			//process_with_gpu(qctx);
#endif
		}
		checked += qctx.checked;
		reached += qctx.found;

		/*
		 *
		 * some statistics printing for debuging only
		 *
		 * */
		map<int, uint> connected;

		uint *gridchecks = pinfo->grid_checkings;
		uint max_one = 0;
		for(int i=0;i<config.num_objects;i++){
			if(connected.find(result[i])==connected.end()){
				connected[result[i]] = 1;
			}else{
				connected[result[i]]++;
			}
			if(result[max_one]<result[i]){
				max_one = i;
			}
		}
		double cum_portion = 0;
		for(auto a:connected){
			cum_portion += 1.0*a.second/config.num_objects;
			printf("%d\t%d\t%f\n",a.first,a.second,cum_portion);
		}

		vector<Point *> all_points;
		vector<Point *> valid_points;
		Point *p1 = cur_trace + max_one;
		for(uint pairid=0;pairid<pinfo->num_grid_checkings;pairid++){
			if(gridchecks[2*pairid]==max_one){
				uint zid = gridchecks[2*pairid+1];
				uint *cur_pid = pinfo->get_zone(zid);
				for(uint i=0;i<pinfo->get_zone_size(zid);i++){
					Point *p2 = cur_trace+cur_pid[i];
					if(p1==p2){
						continue;
					}
					all_points.push_back(p2);
					double dist = p1->distance(*p2,true);
					if(dist<config.reach_distance){
						valid_points.push_back(p2);
					}
				}
			}
		}

		print_points(all_points);
		print_points(valid_points);
		p1->print();
		cout<<all_points.size()<<" "<<valid_points.size()<<endl;
	}

	logt("contact trace with %ld calculation %ld connected",start,checked,reached);
}
