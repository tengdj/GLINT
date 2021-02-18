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
	uint *result = (uint *)ctx->target[1];
	Point *points = pinfo->points;

	checking_unit *grid_check = pinfo->checking_units;
	size_t checked = 0;
	// pick one batch of point-grid pair for processing
	size_t start = 0;
	size_t end = 0;
	while(ctx->next_batch(start,end)){
		for(uint pairid=start;pairid<end;pairid++){
			uint pid = grid_check[pairid].pid;
			uint gid = grid_check[pairid].gid;
			uint size = min(pinfo->get_grid_size(gid)-grid_check[pairid].offset, (uint)pinfo->config.zone_capacity);
			uint *cur_pids = pinfo->get_grid(gid)+grid_check[pairid].offset;
			//vector<Point *> pts;
			Point *p1 = points + pid;
			for(uint i=0;i<size;i++){
				//pts.push_back(points + cur_pids[i]);
				Point *p2 = points + cur_pids[i];
				//p2->print();
				if(p1!=p2){
					//log("%f",dist);
					if(p1->distance(p2, true)<=ctx->config.reach_distance){
						result[pid]++;
					}
					checked++;
				}
			}
		}
	}
	lock();
	ctx->checked += checked;
	unlock();
	return NULL;
}

void process_with_cpu(query_context &tctx){
	struct timeval start = get_cur_time();
	pthread_t threads[tctx.config.num_threads];
	tctx.reset();

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
	part->build_schema(trace, config.num_objects);
	// test contact tracing
	size_t checked = 0;
	size_t reached = 0;
	query_context qctx;
	qctx.config = config;
	for(int t=0;t<config.duration;t++){

		Point *cur_trace = trace+t*config.num_objects;
		part->pinfo->reset();
		part->pinfo->points = cur_trace;
		qctx.target[0] = (void *)part->pinfo;
		qctx.target[1] = (void *)result;
		// process the objects in the packed partitions
		if(!config.gpu){
			part->partition(cur_trace, config.num_objects);
			qctx.num_units = part->pinfo->num_checking_units;
			process_with_cpu(qctx);
		}else{
#ifdef USE_GPU
			process_with_gpu(qctx);
#endif
		}
		for(int i=0;i<config.num_objects;i++){
			qctx.found += result[i];
		}
		checked += qctx.checked;
		reached += qctx.found;

		if(false){
			/*
			 *
			 * some statistics printing for debuging only
			 *
			 * */
			map<int, uint> connected;

			checking_unit *gridchecks = part->pinfo->checking_units;
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
			connected.clear();

//			uint *unit_count = new uint[config.num_objects];
//			memset(unit_count,0,config.num_objects*sizeof(uint));
//			for(uint pairid=0;pairid<pinfo->num_checking_units;pairid++){
//				unit_count[gridchecks[pairid].pid]++;
//			}
//			max_one = 0;
//			for(int i=0;i<config.num_objects;i++){
//				if(unit_count[max_one]<unit_count[i]){
//					max_one = i;
//				}
//			}
//			cout<<max_one<<" "<<unit_count[max_one]<<endl;
//			delete []unit_count;

			vector<Point *> all_points;
			vector<Point *> valid_points;
			Point *p1 = cur_trace + max_one;
			for(uint pairid=0;pairid<part->pinfo->num_checking_units;pairid++){
				if(gridchecks[pairid].pid==max_one&&gridchecks[pairid].offset==0){
					uint *cur_pid = part->pinfo->get_grid(gridchecks[pairid].gid);
					for(uint i=0;i<part->pinfo->get_grid_size(gridchecks[pairid].gid);i++){
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
			cout<<max_one<<" "<<all_points.size()<<" "<<valid_points.size()<<endl;
			all_points.clear();
			valid_points.clear();
		}
	}

	logt("contact trace with %ld calculation %ld connected",start,checked,reached);
}
