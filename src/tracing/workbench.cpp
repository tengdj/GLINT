/*
 * workbench.cpp
 *
 *  Created on: Feb 18, 2021
 *      Author: teng
 */

#include "workbench.h"


workbench::workbench(workbench *bench){
	config = bench->config;

	grid_lookup_capacity = bench->grid_lookup_capacity;
	stack_capacity = bench->stack_capacity;
	reaches_capacity = bench->reaches_capacity;
	meeting_capacity = bench->meeting_capacity;
	meeting_bucket_capacity = bench->meeting_bucket_capacity;
	unit_lookup_capacity = bench->unit_lookup_capacity;
	num_grids = bench->num_grids;
	num_nodes = bench->num_nodes;
	for(int i=0;i<50;i++){
		pthread_mutex_init(&insert_lk[i],NULL);
	}
}

workbench::workbench(configuration *conf){
	config = conf;
	if(config->gpu){
		config->num_objects_per_round = config->num_objects;
	}
	grid_lookup_capacity = 2*config->num_objects;
	stack_capacity = 2*config->num_objects;
	reaches_capacity = 10*config->num_objects;
	meeting_capacity = 10*config->num_objects;
	meeting_bucket_capacity = max((uint)20, reaches_capacity/config->num_meeting_buckets);
	unit_lookup_capacity = config->num_objects_per_round*(config->grid_capacity/config->zone_capacity+1);
	for(int i=0;i<50;i++){
		pthread_mutex_init(&insert_lk[i],NULL);
	}
}

void workbench::clear(){
	if(grids){
		delete []grids;
	}
	if(grid_assignment){
		delete []grid_assignment;
	}
	if(unit_lookup){
		delete []unit_lookup;
	}
	if(schema){
		delete []schema;
	}
	if(reaches){
		delete []reaches;
	}
	if(meeting_buckets){
		delete []meeting_buckets;
	}
	if(meeting_buckets_counter){
		delete []meeting_buckets_counter;
	}
	if(meeting_buckets_counter_tmp){
		delete []meeting_buckets_counter_tmp;
	}
	if(meetings){
		delete []meetings;
	}
}


void workbench::claim_space(uint ng){
		assert(ng>0);

		double total_size = 0;
		if(num_grids != ng){
			if(grids){
				delete []grids;
			}
			num_grids = ng;
			grids = new uint[(config->grid_capacity+1)*num_grids];
			double grid_size = (config->grid_capacity+1)*num_grids*sizeof(uint)/1024.0/1024.0;
			log("\t%.2fMB grids",grid_size);
			total_size += grid_size;
		}

		if(!unit_lookup){
			unit_lookup = new checking_unit[unit_lookup_capacity];
			double cu_size = unit_lookup_capacity*sizeof(checking_unit)/1024.0/1024.0;
			log("\t%.2fMB checking units",cu_size);
		}
		if(!reaches){
			reaches = new reach_unit[reaches_capacity];
			double rc_size = reaches_capacity*sizeof(reach_unit)/1024.0/1024.0;
			log("\t%.2fMB reach space",rc_size);
			total_size += rc_size;
		}
		if(!meeting_buckets){
			meeting_buckets = new meeting_unit[config->num_meeting_buckets*meeting_bucket_capacity];
			double mt_size = config->num_meeting_buckets*meeting_bucket_capacity*sizeof(meeting_unit)/1024.0/1024.0;
			log("\t%.2fMB meeting bucket space",mt_size);
			total_size += mt_size;
		}
		if(!meeting_buckets_counter){
			meeting_buckets_counter = new uint[config->num_meeting_buckets];
			memset(meeting_buckets_counter,0,config->num_meeting_buckets*sizeof(uint));
			meeting_buckets_counter_tmp = new uint[config->num_meeting_buckets];
			memset(meeting_buckets_counter_tmp,0,config->num_meeting_buckets*sizeof(uint));
			double mt_size = 2*config->num_meeting_buckets*sizeof(uint)/1024.0/1024.0;
			log("\t%.2fMB meeting bucket counter space",mt_size);
			total_size += mt_size;
		}
		if(!meetings){
			meetings = new meeting_unit[meeting_capacity];
			double mt_size = meeting_capacity*sizeof(meeting_unit)/1024.0/1024.0;
			log("\t%.2fMB meeting space",mt_size);
			total_size += mt_size;
		}

		log("%.2fMB memory space is claimed",total_size);

	}




bool workbench::insert(uint gid, uint pid){

	lock(gid);
	uint cur_size = grids[(config->grid_capacity+1)*gid];
	if(cur_size>=config->grid_capacity){
		unlock(gid);
		return false;
	}
	assert(cur_size<config->grid_capacity);
	grids[(config->grid_capacity+1)*gid]++;
	grids[(config->grid_capacity+1)*gid+1+cur_size] = pid;
	unlock(gid);
	return true;
}

bool workbench::batch_insert(uint gid, uint num_objects, uint *pids){
	assert(num_objects<config->grid_capacity);
	lock(gid);
	memcpy(grids+(config->grid_capacity+1)*gid+1,pids,num_objects*sizeof(uint));
	*(grids+(config->grid_capacity+1)*gid) = num_objects;
	unlock(gid);
	return true;
}

bool workbench::check(uint gid, uint pid){
	assert(gid<num_grids);
	lock();
	uint offset = 0;
	while(offset<get_grid_size(gid)){
		assert(unit_lookup_counter<unit_lookup_capacity);
		unit_lookup[unit_lookup_counter].pid = pid;
		unit_lookup[unit_lookup_counter].gid = gid;
		unit_lookup[unit_lookup_counter].offset = offset;
		unit_lookup_counter++;
		offset += config->zone_capacity;
	}
	unlock();
	return true;
}

bool workbench::batch_check(checking_unit *buffer, uint num){
	if(num == 0){
		return false;
	}
	uint cur_counter = 0;
	lock();
	assert(unit_lookup_counter+num<unit_lookup_capacity);
	cur_counter = unit_lookup_counter;
	unit_lookup_counter += num;
	unlock();
	memcpy(unit_lookup+cur_counter,buffer,sizeof(checking_unit)*num);
	return true;
}

bool workbench::batch_reach(reach_unit *buffer, uint num){
	if(num == 0){
		return false;
	}
	uint cur_counter = 0;
	lock();
	assert(reaches_counter+num<reaches_capacity);
	cur_counter = reaches_counter;
	reaches_counter += num;
	unlock();
	memcpy(reaches+cur_counter,buffer,sizeof(reach_unit)*num);
	return true;
}


bool workbench::batch_meet(meeting_unit *buffer, uint num){
	if(num == 0){
		return false;
	}
	uint cur_counter = 0;
	assert(meeting_counter+num<meeting_capacity);
	lock();
	cur_counter = meeting_counter;
	meeting_counter += num;
	unlock();
	memcpy(meetings+cur_counter,buffer,sizeof(meeting_unit)*num);
	return true;
}

// single thread function for assigning objects into grids following certain schema
void *partition_unit(void *arg){
	query_context *qctx = (query_context *)arg;
	workbench *bench = (workbench *)qctx->target[0];
	QTSchema *schema = bench->schema;

	// pick one batch of point-grid pair for processing
	size_t start = 0;
	size_t end = 0;
	while(qctx->next_batch(start,end)){
		for(uint pid=start;pid<end;pid++){
			uint gid = 0;
			uint curoff = 0;
			Point *p = bench->points+pid;
			while(true){
				int loc = (p->y>bench->schema[curoff].mid_y)*2+(p->x>bench->schema[curoff].mid_x);
				uint child_offset = bench->schema[curoff].children[loc];
				// is leaf
				if(bench->schema[child_offset].isleaf){
					gid = bench->schema[child_offset].node_id;
					break;
				}
				curoff = child_offset;
			}
			bench->insert(gid, pid);
		}
	}
	return NULL;
}



void workbench::partition(){
	// the schema has to be built
	struct timeval start = get_cur_time();

	// partitioning current batch of objects with the existing schema
	pthread_t threads[config->num_threads];
	query_context qctx;
	qctx.config = config;
	qctx.num_units = config->num_objects;
	qctx.target[0] = (void *)this;

	for(int i=0;i<config->num_threads;i++){
		pthread_create(&threads[i], NULL, partition_unit, (void *)&qctx);
	}
	for(int i = 0; i < config->num_threads; i++ ){
		void *status;
		pthread_join(threads[i], &status);
	}
	logt("partition",start);
}


/*
 *
 * the CPU functions for looking up QTree with points and generate pid-gid-offset
 *
 * */


void lookup_rec(QTSchema *schema, Point *p, uint curoff, vector<uint> &gids, double x_buffer, double y_buffer){

	// could be possibly in multiple children with buffers enabled
	bool top = (p->y>schema[curoff].mid_y-y_buffer);
	bool bottom = (p->y<=schema[curoff].mid_y+y_buffer);
	bool left = (p->x<=schema[curoff].mid_x+x_buffer);
	bool right = (p->x>schema[curoff].mid_x-x_buffer);
	bool need_check[4] = {bottom&&left, bottom&&right, top&&left, top&&right};
	for(int i=0;i<4;i++){
		if(need_check[i]){
			uint child_offset = schema[curoff].children[i];
			if(schema[child_offset].isleaf){
				gids.push_back(schema[child_offset].node_id);
			}else{
				lookup_rec(schema, p, child_offset, gids, x_buffer, y_buffer);
			}
		}
	}
}

// single thread function for looking up the schema to generate point-grid pairs for processing
void *lookup_unit(void *arg){
	query_context *qctx = (query_context *)arg;
	workbench *bench = (workbench *)qctx->target[0];
	QTSchema *schema = bench->schema;

	// pick one point for looking up
	size_t start = 0;
	size_t end = 0;
	vector<uint> gids;
	checking_unit *cubuffer = new checking_unit[200];
	uint buffer_index = 0;
	while(qctx->next_batch(start,end)){
		for(uint pid=start;pid<end;pid++){
			Point *p = bench->points+pid;
			lookup_rec(schema, p, 0, gids, qctx->config->x_buffer, qctx->config->y_buffer);
			for(uint gid:gids){
				assert(gid<bench->num_grids);
				uint offset = 0;
				while(offset<bench->get_grid_size(gid)){

					cubuffer[buffer_index].pid = pid;
					cubuffer[buffer_index].gid = gid;
					cubuffer[buffer_index].offset = offset;
					buffer_index++;
					if(buffer_index==200){
						bench->batch_check(cubuffer, buffer_index);
						buffer_index = 0;
					}
					offset += bench->config->zone_capacity;
				}
			}
			gids.clear();
		}
	}
	bench->batch_check(cubuffer, buffer_index);
	delete []cubuffer;
	return NULL;
}

void workbench::lookup(uint start_pid){
	struct timeval start = get_cur_time();
	// reset the counter
	unit_lookup_counter = 0;
	// partitioning current batch of objects with the existing schema
	pthread_t threads[config->num_threads];
	query_context qctx;
	qctx.config = config;
	qctx.counter = start_pid;
	qctx.num_units = min(config->num_objects, start_pid+config->num_objects_per_round);
	qctx.target[0] = (void *)this;

	// tree lookups
	for(int i=0;i<config->num_threads;i++){
		pthread_create(&threads[i], NULL, lookup_unit, (void *)&qctx);
	}
	for(int i = 0; i < config->num_threads; i++ ){
		void *status;
		pthread_join(threads[i], &status);
	}
	logt("lookup",start);
}




void *reachability_unit(void *arg){
	query_context *ctx = (query_context *)arg;
	workbench *bench = (workbench *)ctx->target[0];
	Point *points = bench->points;
	reach_unit *reach_buffer = new reach_unit[200];
	uint reach_index = 0;

	// pick one batch of point-grid pair for processing
	size_t start = 0;
	size_t end = 0;
	while(ctx->next_batch(start,end)){
		for(uint pairid=start;pairid<end;pairid++){
			uint pid = bench->unit_lookup[pairid].pid;
			uint gid = bench->unit_lookup[pairid].gid;
			uint offset = bench->unit_lookup[pairid].offset;

			uint size = min(bench->get_grid_size(gid)-offset, (uint)bench->config->zone_capacity);
			uint *cur_pids = bench->get_grid(gid)+offset;

			//vector<Point *> pts;
			Point *p1 = points + pid;
			for(uint i=0;i<size;i++){
				//pts.push_back(points + cur_pids[i]);
				Point *p2 = points + cur_pids[i];
				//p2->print();
				if(p1!=p2 && p1->distance(p2, true)<=ctx->config->reach_distance){
					reach_buffer[reach_index].pid1 = pid;
					reach_buffer[reach_index].pid2 = cur_pids[i];
					if(++reach_index==200){
						bench->batch_reach(reach_buffer,reach_index);
						reach_index = 0;
					}
				}
			}
		}
	}

	bench->batch_reach(reach_buffer,reach_index);
	delete []reach_buffer;
	return NULL;
}

void workbench::reachability(){

	query_context tctx;
	tctx.config = config;
	tctx.num_units = unit_lookup_counter;
	tctx.target[0] = (void *)this;

	// generate a new batch of reaches
	reaches_counter = 0;
	struct timeval start = get_cur_time();
	pthread_t threads[tctx.config->num_threads];

	for(int i=0;i<tctx.config->num_threads;i++){
		pthread_create(&threads[i], NULL, reachability_unit, (void *)&tctx);
	}
	for(int i = 0; i < tctx.config->num_threads; i++ ){
		void *status;
		pthread_join(threads[i], &status);
	}
	//bench->unit_lookup_counter = 0;
	logt("compute",start);
}


/*
 *
 * update the meetings maintained with reachability information collected in this round
 *
 * */


void *update_meetings_unit(void *arg){
	query_context *ctx = (query_context *)arg;
	workbench *bench = (workbench *)ctx->target[0];

	size_t start = 0;
	size_t end = 0;
	while(ctx->next_batch(start,end)){
		for(uint rid=start;rid<end;rid++){
			uint pid1 = bench->reaches[rid].pid1;
			uint pid2 = bench->reaches[rid].pid2;
			uint bid = (pid1+pid2)%bench->config->num_meeting_buckets;
			meeting_unit *bucket = bench->meeting_buckets+bid*bench->meeting_bucket_capacity;
			bool updated = false;
			for(uint i=0;i<bench->meeting_buckets_counter_tmp[bid];i++){
				// a former meeting is encountered, update it
				if(bucket[i].pid1==pid1&&bucket[i].pid2==pid2){
					bucket[i].end = bench->cur_time;
					updated = true;
					break;
				}
			}
			if(!updated){
				uint loc = 0;
				bench->lock(bid);
				loc = bench->meeting_buckets_counter[bid]++;
				bench->unlock(bid);
				assert(loc<bench->meeting_bucket_capacity);
				bucket[loc].pid1 = pid1;
				bucket[loc].pid2 = pid2;
				bucket[loc].start = bench->cur_time;
				bucket[loc].end = bench->cur_time;
			}
		}
	}
	return NULL;
}

void workbench::update_meetings(){
	struct timeval start = get_cur_time();

	query_context tctx;
	tctx.config = config;
	tctx.target[0] = (void *)this;
	tctx.num_units = reaches_counter;
	pthread_t threads[tctx.config->num_threads];

	for(int i=0;i<tctx.config->num_threads;i++){
		pthread_create(&threads[i], NULL, update_meetings_unit, (void *)&tctx);
	}
	for(int i = 0; i < tctx.config->num_threads; i++ ){
		void *status;
		pthread_join(threads[i], &status);
	}
	logt("update meetings",start);
}



void *compact_meetings_unit(void *arg){
	query_context *ctx = (query_context *)arg;
	workbench *bench = (workbench *)ctx->target[0];

	size_t start = 0;
	size_t end = 0;
	meeting_unit *mu_buffer = new meeting_unit[200];
	uint mu_index = 0;

	while(ctx->next_batch(start,end)){
		for(uint bid=start;bid<end;bid++){
			meeting_unit *bucket = bench->meeting_buckets+bid*bench->meeting_bucket_capacity;
			int front_idx = 0;
			int back_idx = bench->meeting_buckets_counter[bid]-1;
			int active_count = 0;
			for(;front_idx<=back_idx;front_idx++){
				// this meeting is over
				if(bucket[front_idx].end<bench->cur_time){
					// dump to valid list and copy one from the back end
					if(bucket[front_idx].end-bucket[front_idx].start>=bench->config->min_meet_time){
						mu_buffer[mu_index++] = bucket[front_idx];
						if(mu_index==200){
							bench->batch_meet(mu_buffer,mu_index);
							mu_index = 0;
						}
					}
					for(;back_idx>front_idx;back_idx--){
						if(bucket[back_idx].end==bench->cur_time){
							break;
							// dump to valid list if needed or disregarded
						}else if(bucket[back_idx].end-bucket[back_idx].start>=bench->config->min_meet_time){
							mu_buffer[mu_index++] = bucket[back_idx];
							if(mu_index==200){
								bench->batch_meet(mu_buffer,mu_index);
								mu_index = 0;
							}
						}
					}
					if(front_idx<back_idx){
						bucket[front_idx] = bucket[back_idx];
						active_count++;
						back_idx--;
					}
				}else{
					active_count++;
				}
			}

			bench->meeting_buckets_counter[bid] = active_count;
			bench->meeting_buckets_counter_tmp[bid] = active_count;
		}
	}
	bench->batch_meet(mu_buffer,mu_index);
	delete []mu_buffer;
	return NULL;
}

void workbench::compact_meetings(){
	struct timeval start = get_cur_time();

	query_context tctx;
	tctx.config = config;
	tctx.target[0] = (void *)this;
	tctx.num_units = config->num_meeting_buckets;
	pthread_t threads[tctx.config->num_threads];

	for(int i=0;i<tctx.config->num_threads;i++){
		pthread_create(&threads[i], NULL, compact_meetings_unit, (void *)&tctx);
	}
	for(int i = 0; i < tctx.config->num_threads; i++ ){
		void *status;
		pthread_join(threads[i], &status);
	}
	logt("%d meetings compacted",start,this->meeting_counter);

}



void workbench::analyze_grids(){

}

void workbench::analyze_checkings(){

}


void workbench::analyze_meetings(){

	/*
	 *
	 * some statistics printing for debuging only
	 *
	 * */

	uint *unit_count = new uint[config->num_objects];
	memset(unit_count,0,config->num_objects*sizeof(uint));
	uint min_bucket = 0;
	uint max_bucket = 0;
	for(uint i=0;i<config->num_meeting_buckets;i++){
		if(meeting_buckets_counter[min_bucket]>meeting_buckets_counter[i]){
			min_bucket = i;
		}
		if(meeting_buckets_counter[max_bucket]<meeting_buckets_counter[i]){
			max_bucket = i;
		}
		meeting_unit *bucket = meeting_buckets+i*this->meeting_bucket_capacity;
		for(uint j=0;j<meeting_buckets_counter[i];j++){
			unit_count[bucket[j].pid1]++;
			if(bucket[j].pid1==0){
				printf("%d %d %d\n",bucket[j].pid2,bucket[j].start,bucket[j].end);
			}
		}
	}
	log("bucket range: [%d, %d]",meeting_buckets_counter[min_bucket],meeting_buckets_counter[max_bucket]);
	map<int, uint> connected;
	uint max_one = 0;
	for(int i=0;i<config->num_objects;i++){
		if(connected.find(unit_count[i])==connected.end()){
			connected[unit_count[i]] = 1;
		}else{
			connected[unit_count[i]]++;
		}
		if(unit_count[max_one]<unit_count[i]){
			max_one = i;
		}
	}
	log("%d contains max objects",max_one);
	double cum_portion = 0;
	for(auto a:connected){
		cum_portion += 1.0*a.second/config->num_objects;
		log("%d\t%d\t%f",a.first,a.second,cum_portion);
	}
	connected.clear();

	vector<Point *> all_points;
	vector<Point *> valid_points;
	Point *p1 = points + max_one;
	vector<uint> gids;
	lookup_rec(schema, p1, 0, gids, config->x_buffer, config->y_buffer);

	for(uint gid:gids){
		uint *cur_pid = get_grid(gid);
		for(uint i=0;i<get_grid_size(gid);i++){
			Point *p2 = points+cur_pid[i];
			if(p1==p2){
				continue;
			}
			all_points.push_back(p2);
			double dist = p1->distance(*p2,true);
			if(dist<config->reach_distance){
				valid_points.push_back(p2);
			}
		}
	}


	log("point %d has %d contacts in result, %ld checked, %ld validated"
			,max_one,unit_count[max_one],all_points.size(), valid_points.size());
	print_points(all_points);
	print_points(valid_points);
	p1->print();
	all_points.clear();
	valid_points.clear();
	delete []unit_count;

}



