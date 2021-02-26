/*
 * trace_generator.cpp
 *
 *  Created on: Feb 1, 2021
 *      Author: teng
 */

#include "../index/QTree.h"
#include "trace.h"

/*
 *
 * Trip member functions
 *
 * */

Trip::Trip(string str){

	vector<string> cols;
	tokenize(str,cols,",");

	start.timestamp = 0;
	char tmp[2];
	tmp[0] = cols[2][11];
	tmp[1] = cols[2][12];
	start.timestamp += atoi(tmp)*3600;
	tmp[0] = cols[2][14];
	tmp[1] = cols[2][15];
	start.timestamp += atoi(tmp)*60;
	tmp[0] = cols[2][17];
	tmp[1] = cols[2][18];
	start.timestamp += atoi(tmp);
	if(cols[2][20]=='P'){
		start.timestamp += 12*3600;
	}
	end.timestamp = start.timestamp + atoi(cols[4].c_str());

	start.coordinate = Point(atof(cols[18].c_str()),atof(cols[17].c_str()));
	end.coordinate = Point(atof(cols[21].c_str()),atof(cols[20].c_str()));
}

void Trip::print_trip(){
	printf("time: %d to %d\n",start.timestamp,end.timestamp);
	printf("position: (%f %f) to (%f %f)\n",start.coordinate.x,start.coordinate.y,end.coordinate.x,end.coordinate.y);
}

void Trip::resize(int md){
	if(md>0&&duration()>md){
		if(type==REST){
			end.timestamp = start.timestamp+md;
		}else{
			double portion = md*1.0/duration();
			end.coordinate.x = (end.coordinate.x-start.coordinate.x)*portion+start.coordinate.x;
			end.coordinate.y = (end.coordinate.y-start.coordinate.y)*portion+start.coordinate.y;
			end.timestamp = start.timestamp + md+1;
		}
	}
}


/*
 *
 * functions for generating simulated traces of an object
 * based on the real world statistics
 *
 * */

bool orderzone(ZoneStats *i, ZoneStats *j) { return (i->count>j->count); }

void trace_generator::analyze_trips(const char *path, int limit){
	struct timeval start = get_cur_time();

	if(total){
		delete total;
	}
	std::ifstream file(path);
	std::string str;
	//skip the head
	std::getline(file, str);
	total = new ZoneStats(0);

	while (std::getline(file, str)&&--limit>0){
		Trip *t = new Trip(str);
		t->start.coordinate.x += 0.02*(get_rand_double()-0.5);
		t->start.coordinate.y += 0.012*(get_rand_double()-0.5);
		t->end.coordinate.x += 0.02*(get_rand_double()-0.5);
		t->end.coordinate.y += 0.012*(get_rand_double()-0.5);
		// a valid trip should be covered by the map,
		// last for a while and the distance larger than 0
		if(map->getMBR()->contain(t->start.coordinate)&&
		   map->getMBR()->contain(t->end.coordinate)&&
		   t->length()>0&&
		   t->duration()>0){

			int gids[2];
			gids[0] = grid->getgridid(&t->start.coordinate);
			gids[1] = grid->getgridid(&t->end.coordinate);
			double dist = t->start.coordinate.distance(t->end.coordinate, true);
			for(int i=0;i<2;i++){
				int zid = gids[i];
				int ezid = gids[!i];
				zones[zid]->count++;
				zones[zid]->duration += t->duration();
				zones[zid]->length += dist;
				total->count++;
				total->duration += t->duration();
				total->length += dist;
			}
		}
		delete t;
	}
	file.close();

	// reorganize
	sort(zones.begin(),zones.end(),orderzone);
	logt("analyze trips in %s",start, path);
}


/*
 *
 * get the next location according to the distribution of the statistic
 *
 *
 * */


Point trace_generator::get_random_location(){
	int start_x = -1;
	int start_y = -1;
	// certain portion follows the distribution
	// of analyzed dataset, the rest randomly generate
	if(tryluck(1.0-config->distribution_rate)){
		double target = get_rand_double();
		//log("%f",target);
		double cum = 0;
		for(ZoneStats *z:zones){
			//log("%d",z->count);
			double next_cum = cum + z->count*1.0/total->count;
			if(target>=cum&&target<=next_cum){
				start_x = z->zoneid%grid->dimx;
				start_y = z->zoneid/grid->dimx;
				break;
			}
			cum = next_cum;
		}
		assert(start_x>=0);
	}
	return grid->get_random_point(start_x, start_y);
}

/*
 * simulate next move.
 *
 * */

Trip *trace_generator::next_trip(Trip *former){
	Trip *next = new Trip();
	// get a start location if no previous trip
	if(former==NULL){
		next->start.coordinate = get_random_location();
		next->start.timestamp = 0;
	}else{
		next->start = former->end;
	}

	// rest for a while until the end, the time is adjustable
	if(tryluck(config->walk_rate)){
		next->type = WALK;
		next->end.coordinate = get_random_location();
		next->end.timestamp = next->start.timestamp+next->end.coordinate.distance(next->start.coordinate, true)/config->walk_speed;
	}else if(tryluck(config->drive_rate)){
		next->type = DRIVE;
		next->end.coordinate = get_random_location();
		next->end.timestamp = next->start.timestamp+(next->end.coordinate.distance(next->start.coordinate, true))/config->drive_speed+1;
	}else{
		next->end = next->start;
		next->type = REST;
	}
	return next;
}


vector<Point *> trace_generator::get_trace(Map *mymap){
	// use the default map for single thread mode
	if(!mymap){
		mymap = map;
	}
	assert(mymap);
	vector<Point *> ret;
	Trip *trip = next_trip();
	trip->resize(config->duration);
	Point *first_point = new Point(trip->start.coordinate.x,trip->start.coordinate.y);
	ret.push_back(first_point);
	while(ret.size()<config->duration){
		// stay here
		if(trip->type==REST){
			for(int i=0;i<trip->duration()&&ret.size()<config->duration;i++){
				ret.push_back(new Point(&trip->start.coordinate));
			}
		}else if(trip->type==WALK){ //walk
			const double step = 1.0/trip->duration();
			double portion = 0;
			for(int i=0;i<trip->duration()&&ret.size()<config->duration;i++){
				double px = trip->start.coordinate.x+portion*(trip->end.coordinate.x - trip->start.coordinate.x);
				double py = trip->start.coordinate.y+portion*(trip->end.coordinate.y - trip->start.coordinate.y);
				ret.push_back(new Point(px,py));
				portion += step;
			}
		}else{//drive
			mymap->navigate(ret, &trip->start.coordinate, &trip->end.coordinate, trip->speed());
		}
		// move to another trip following last trip
		Trip *newtrip = next_trip(trip);
		delete trip;
		trip = newtrip;
		trip->resize(config->duration-ret.size());
	}
	for(int i=config->duration;i<ret.size();i++){
		delete ret[i];
	}
	ret.erase(ret.begin()+config->duration,ret.end());
	assert(ret.size()==config->duration);
	delete trip;
	return ret;
}

void *gentrace_unit(void *arg){
	query_context *ctx = (query_context *)arg;
	trace_generator *gen = (trace_generator *)ctx->target[0];
	Point *result = (Point *)ctx->target[1];
	Map *mymap = gen->map->clone();
	while(true){
		// pick one object for generating
		size_t start = 0;
		size_t end = 0;
		if(!ctx->next_batch(start,end)){
			break;
		}
		for(int obj=start;obj<end;obj++){
			//log("%d",obj);
			vector<Point *> trace = gen->get_trace(mymap);
			// copy to target
			for(int i=0;i<gen->config->duration;i++){
				result[i*gen->config->num_objects+obj] = *trace[i];
				delete trace[i];
			}
			trace.clear();
		}
	}
	delete mymap;
	return NULL;
}


Point *trace_generator::generate_trace(){
	struct timeval start = get_cur_time();
	Point *ret = (Point *)malloc(config->duration*config->num_objects*sizeof(Point));
	pthread_t threads[config->num_threads];
	query_context tctx;
	tctx.config = config;
	tctx.target[0] = (void *)this;
	tctx.target[1] = (void *)ret;
	tctx.num_units = config->num_objects;
	tctx.report_gap = 1;
	for(int i=0;i<config->num_threads;i++){
		pthread_create(&threads[i], NULL, gentrace_unit, (void *)&tctx);
	}
	for(int i = 0; i < config->num_threads; i++ ){
		void *status;
		pthread_join(threads[i], &status);
	}
	logt("generate traces",start);
	return ret;
}
