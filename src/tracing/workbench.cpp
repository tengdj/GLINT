/*
 * workbench.cpp
 *
 *  Created on: Feb 18, 2021
 *      Author: teng
 */

#include "workbench.h"


workbench::workbench(workbench *bench):workbench(bench->config){
	grids_counter = bench->grids_counter;
	schema_counter = bench->schema_counter;
}

workbench::workbench(configuration *conf){
	config = conf;

	// setup the capacity of each container

	// each grid contains averagely grid_capacity/2 objects, times 3 for enough space
	grids_capacity = 3*max((uint)1, config->num_objects/config->grid_capacity);
	// the number of all QTree Nodes
	schema_capacity = 2*grids_capacity;
	stack_capacity = 2*config->num_objects;
	reaches_capacity = 10*config->num_objects;
	meeting_capacity = 10*config->num_objects;
	meeting_bucket_capacity = max((uint)20, reaches_capacity/config->num_meeting_buckets);
	grid_check_capacity = config->num_objects*(config->grid_capacity/config->zone_capacity+1);

	for(int i=0;i<50;i++){
		pthread_mutex_init(&insert_lk[i],NULL);
	}
}

void workbench::clear(){
	if(grids){
		delete []grids;
	}
	if(grid_counter){
		delete []grid_counter;
	}
	if(grid_check){
		delete []grid_check;
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
	if(lookup_stack[0]){
		delete lookup_stack[0];
	}
	if(lookup_stack[1]){
		delete lookup_stack[1];
	}
}


void workbench::claim_space(){

	struct timeval start = get_cur_time();

	double total_size = 0;
	double tmp_size = 0;

	grids = new uint[config->grid_capacity*grids_capacity];
	tmp_size = config->grid_capacity*grids_capacity*sizeof(uint)/1024.0/1024.0;
	log("\t%.2f MB\tgrids",tmp_size);
	total_size += tmp_size;

	grid_counter = new uint[grids_capacity];
	tmp_size = grids_capacity*sizeof(uint)/1024.0/1024.0;
	log("\t%.2f MB\tgrid counter",tmp_size);
	total_size += tmp_size;

	schema = new QTSchema[2*grids_capacity];
	tmp_size = 2*grids_capacity*sizeof(QTSchema)/1024.0/1024.0;
	log("\t%.2f MB  \tschema",tmp_size);
	total_size += tmp_size;

	grid_check = new checking_unit[grid_check_capacity];
	tmp_size = grid_check_capacity*sizeof(checking_unit)/1024.0/1024.0;
	log("\t%.2f MB\tchecking units",tmp_size);
	total_size += tmp_size;

	reaches = new reach_unit[reaches_capacity];
	tmp_size = reaches_capacity*sizeof(reach_unit)/1024.0/1024.0;
	log("\t%.2f MB\treach space",tmp_size);
	total_size += tmp_size;

	meeting_buckets = new meeting_unit[config->num_meeting_buckets*meeting_bucket_capacity];
	tmp_size = config->num_meeting_buckets*meeting_bucket_capacity*sizeof(meeting_unit)/1024.0/1024.0;
	log("\t%.2f MB\tmeeting bucket space",tmp_size);
	total_size += tmp_size;

	meeting_buckets_counter = new uint[config->num_meeting_buckets];
	memset(meeting_buckets_counter,0,config->num_meeting_buckets*sizeof(uint));
	meeting_buckets_counter_tmp = new uint[config->num_meeting_buckets];
	memset(meeting_buckets_counter_tmp,0,config->num_meeting_buckets*sizeof(uint));
	tmp_size = 2*config->num_meeting_buckets*sizeof(uint)/1024.0/1024.0;
	log("\t%.2f MB  \tmeeting bucket counter space",tmp_size);
	total_size += tmp_size;

	meetings = new meeting_unit[meeting_capacity];
	tmp_size = meeting_capacity*sizeof(meeting_unit)/1024.0/1024.0;
	log("\t%.2f MB\tmeeting space",tmp_size);
	total_size += tmp_size;

	lookup_stack[0] = new uint[2*stack_capacity];
	lookup_stack[1] = new uint[2*stack_capacity];
	tmp_size = 2*stack_capacity*2*sizeof(uint)/1024.0/1024.0;
	log("\t%.2f MB\tstack space",tmp_size);
	total_size += tmp_size;

	logt("%.2f MB memory space is claimed",start,total_size);
}


bool workbench::insert(uint curnode, uint pid){
	assert(schema[curnode].isleaf);
	uint gid = schema[curnode].node_id;
	lock(gid);
	uint cur_size = grid_counter[gid]++;
	// todo handle overflow
	if(cur_size<config->grid_capacity){
		grids[config->grid_capacity*gid+cur_size] = pid;
	}
	unlock(gid);
	// first batch of lookup pairs, start from offset 0
	grid_check[pid].pid = pid;
	grid_check[pid].gid = gid;
	grid_check[pid].offset = 0;
	grid_check[pid].inside = true;

	// is this point too close to the border?
	Point *p = points+pid;
	if(p->x+config->x_buffer>schema[curnode].mbr.high[0]||
	   p->y+config->y_buffer>schema[curnode].mbr.high[1]){
		lock();
		lookup_stack[0][stack_index[0]*2] = pid;
		lookup_stack[0][stack_index[0]*2+1] = 0;
		stack_index[0]++;
		unlock();
	}
	return true;
}

//bool workbench::batch_insert(uint gid, uint num_objects, uint *pids){
//	assert(num_objects<config->grid_capacity);
//	lock(gid);
//	memcpy(grids+(config->grid_capacity+1)*gid+1,pids,num_objects*sizeof(uint));
//	*(grids+(config->grid_capacity+1)*gid) = num_objects;
//	unlock(gid);
//	return true;
//}

bool workbench::check(uint gid, uint pid){
	assert(gid<grids_counter);
	lock();
	uint offset = 0;
	while(offset<get_grid_size(gid)){
		assert(grid_check_counter<grid_check_capacity);
		grid_check[grid_check_counter].pid = pid;
		grid_check[grid_check_counter].gid = gid;
		grid_check[grid_check_counter].offset = offset;
		grid_check_counter++;
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
	assert(grid_check_counter+num<grid_check_capacity);
	cur_counter = grid_check_counter;
	grid_check_counter += num;
	unlock();
	memcpy(grid_check+cur_counter,buffer,sizeof(checking_unit)*num);
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

	// pick one batch of point-grid pair for processing
	size_t start = 0;
	size_t end = 0;
	while(qctx->next_batch(start,end)){
		for(uint pid=start;pid<end;pid++){
			uint curnode = 0;
			Point *p = bench->points+pid;
			while(true){
				// assign to a child 0-3
				int child = (p->y>bench->schema[curnode].mid_y)*2+(p->x>bench->schema[curnode].mid_x);
				curnode = bench->schema[curnode].children[child];
				// is leaf
				if(bench->schema[curnode].isleaf){
					break;
				}
			}
			// pid belongs to such node
			bench->insert(curnode, pid);
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
	grid_check_counter = config->num_objects;
	logt("partition data: %d boundary points", start,stack_index[0]);
}

/*
 *
 * the CPU functions for looking up QTree with points
 *
 * */

void lookup_rec(QTSchema *schema, Point *p, uint curnode, vector<uint> &nodes, double max_dist, bool query_all){

	int cur = schema[curnode].which(p);
	// could be possibly in multiple children with buffers enabled
	for(int i=cur;i<4;i++){
		uint child_offset = schema[curnode].children[i];
		double dist = schema[child_offset].mbr.distance(*p, true);
		if(dist<=max_dist){
			if(schema[child_offset].isleaf){
				//if(query_all||i>cur)
				{
					nodes.push_back(child_offset);
				}
			}else{
				lookup_rec(schema, p, child_offset, nodes, max_dist, query_all);
			}
		}
	}
}

// single thread function for looking up the schema to generate point-grid pairs for processing
void *lookup_unit(void *arg){
	query_context *qctx = (query_context *)arg;
	workbench *bench = (workbench *)qctx->target[0];

	// pick one point for looking up
	size_t start = 0;
	size_t end = 0;
	vector<uint> nodes;
	checking_unit *cubuffer = new checking_unit[2000];
	uint buffer_index = 0;
	while(qctx->next_batch(start,end)){
		for(uint sid=start;sid<end;sid++){
			uint pid = bench->lookup_stack[0][2*sid];
			Point *p = bench->points+pid;
			lookup_rec(bench->schema, p, 0, nodes, qctx->config->reach_distance);
			//log("%d",nodes.size());
			for(uint n:nodes){
				uint gid = bench->schema[n].node_id;
				assert(gid<bench->grids_counter);
				cubuffer[buffer_index].pid = pid;
				cubuffer[buffer_index].gid = gid;
				cubuffer[buffer_index].offset = 0;
				cubuffer[buffer_index].inside = false;
				buffer_index++;
				if(buffer_index==2000){
					bench->batch_check(cubuffer, buffer_index);
					buffer_index = 0;
				}
			}
			nodes.clear();
		}
	}
	bench->batch_check(cubuffer, buffer_index);
	delete []cubuffer;
	return NULL;
}

void workbench::lookup(){
	struct timeval start = get_cur_time();
	// partitioning current batch of objects with the existing schema
	pthread_t threads[config->num_threads];
	query_context qctx;
	qctx.config = config;
	qctx.num_units = stack_index[0];
	qctx.target[0] = (void *)this;

	// tree lookups
	for(int i=0;i<config->num_threads;i++){
		pthread_create(&threads[i], NULL, lookup_unit, (void *)&qctx);
	}
	for(int i = 0; i < config->num_threads; i++ ){
		void *status;
		pthread_join(threads[i], &status);
	}
	logt("lookup: %d pid-gid pairs need be checked",start,grid_check_counter);
}




void *reachability_unit(void *arg){
	query_context *ctx = (query_context *)arg;
	workbench *bench = (workbench *)ctx->target[0];
	Point *points = bench->points;
	reach_unit *reach_buffer = new reach_unit[2000];
	uint reach_index = 0;

	// pick one batch of point-grid pair for processing
	size_t start = 0;
	size_t end = 0;
	while(ctx->next_batch(start,end)){
		for(uint pairid=start;pairid<end;pairid++){
			uint pid = bench->grid_check[pairid].pid;
			uint gid = bench->grid_check[pairid].gid;
			uint offset = bench->grid_check[pairid].offset;

			uint size = min(bench->get_grid_size(gid)-offset, (uint)bench->config->zone_capacity);
			uint *cur_pids = bench->get_grid(gid)+offset;

			//vector<Point *> pts;
			Point *p1 = points + pid;
			for(uint i=0;i<size;i++){
				//pts.push_back(points + cur_pids[i]);
				if(pid<cur_pids[i]||!bench->grid_check[pairid].inside){
					Point *p2 = points + cur_pids[i];
					//p2->print();
					if(p1->distance(p2, true)<=ctx->config->reach_distance){
						reach_buffer[reach_index].pid1 = min(pid,cur_pids[i]);
						reach_buffer[reach_index].pid2 = max(cur_pids[i],pid);
						if(++reach_index==2000){
							bench->batch_reach(reach_buffer,reach_index);
							reach_index = 0;
						}
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
	tctx.num_units = grid_check_counter;
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
	//bench->grid_check_counter = 0;
	logt("reachability compute: %d reaches are found",start,reaches_counter);
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
	int mc = 0;
	for(int i=0;i<config->num_meeting_buckets;i++){
		mc += meeting_buckets_counter[i];
	}
	logt("update meeting: %d meetings active",start,mc);
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
	logt("compact meeting: %d meetings recorded",start,meeting_counter);

}

