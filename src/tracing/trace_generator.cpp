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


/*
 *
 * functions for generating simulated traces of an object
 * based on the real world statistics
 *
 * */

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
		// a valid trip should be covered by the map,
		// last for a while and the distance larger than 0
		if(map->getMBR()->contain(t->start.coordinate)&&
		   map->getMBR()->contain(t->end.coordinate)&&
		   t->length()>0&&
		   t->duration()>0){
			int zid = grid->getgridid(&t->start.coordinate);
			zones[zid]->count++;
			zones[zid]->duration += t->duration();
			double dist = t->start.coordinate.distance(t->end.coordinate, true);
			zones[zid]->length += dist;
			total->count++;
			total->duration += t->duration();
			total->length += dist;

			int ezid = grid->getgridid(&t->end.coordinate);
			if(zones[zid]->target_count.find(ezid)==zones[zid]->target_count.end()){
				zones[zid]->target_count[ezid] = 1;
			}else{
				zones[zid]->target_count[ezid]++;
			}
		}
		delete t;
	}
	file.close();
	for(int y=0;y<grid->dimy;y++){
		for(int x=0;x<=grid->dimx;x++){
			int zid = y*grid->dimx+x;
			assert(zid<=grid->get_grid_num());
			if(zones[zid]->count==0){
				zones[zid]->count = 1;
				zones[zid]->duration = total->duration/total->count;
				zones[zid]->length = total->length/total->count;
			}
			assert(zones[zid]->length/zones[zid]->duration>0);
			//printf("%.3f\t",zones[zid]->length*1000.0/zones[zid]->duration);
		}
		//printf("\n");
	}
	logt("analyze trips in %s",start, path);
}

/*
 * simulate next move.
 *
 * */

int inside = 0;
int outside = 0;
Trip *trace_generator::next_trip(Trip *former){
	Trip *next = new Trip();
	if(former==NULL){
		do{
			next->start.timestamp = 0;
			int start_x = -1;
			int start_y = -1;
			do{
				for(int i=0;i<=grid->dimx&&start_x<0;i++){
					for(int j=0;j<grid->dimy;j++){
						if(zones[j*grid->dimx+i]->count>1&&tryluck(zones[j*grid->dimx+i]->count*1.0/total->count)){
							start_x = i;
							start_y = j;
							break;
						}
					}
				}
			}while(start_x<0);
			next->start.coordinate = grid->get_random_point(start_x, start_y);
		}while(false);
	}else{
		next->start = former->end;
	}

	// now generate the next destination according to current position
	int locstart = grid->getgridid(&next->start.coordinate);
	double rest = get_rand_double()*zones[locstart]->rate_sleep;
	// rest for a while, the time is adjustable
	if(tryluck(rest)){
		assert(false&&"invalid now");
		next->end = next->start;
		next->end.timestamp += zones[locstart]->max_sleep_time*rest/zones[locstart]->rate_sleep;
	}else{
		int dest = -1;
		for (auto const& t: zones[locstart]->target_count){
			if(tryluck((double)t.second/zones[locstart]->count)){
				dest = t.first;
				break;
			}
		}
		if(dest == -1){
			dest = get_rand_number(grid->dimx*grid->dimy);
		}
		int x = dest%grid->dimx;
		int y = dest/grid->dimx;


		next->end.coordinate = grid->get_random_point(x, y);
		next->end.timestamp = next->start.timestamp+next->length(true)/zones[locstart]->get_speed();
		int gid = grid->getgridid(&next->end.coordinate);
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
	while(ret.size()<config.duration){
		//trip->print_trip();
		// stay here
		if(trip->start.coordinate.equals(trip->end.coordinate)){
			for(int i=0;i<trip->duration()&&ret.size()<config.duration;i++){
				ret.push_back(new Point(&trip->start.coordinate));
			}
		}else{
			double speed = trip->end.coordinate.distance(trip->start.coordinate, true)/trip->duration();
			mymap->navigate(ret, &trip->start.coordinate, &trip->end.coordinate, speed, config.duration);
		}
		// move to another
		Trip *newtrip = next_trip(trip);
		delete trip;
		trip = newtrip;
	}
	for(int i=config.duration;i<ret.size();i++){
		delete ret[i];
	}
	ret.erase(ret.begin()+config.duration,ret.end());
	assert(ret.size()==config.duration);
	delete trip;
	return ret;
}

void *gentrace(void *arg){
	query_context *ctx = (query_context *)arg;
	trace_generator *gen = (trace_generator *)ctx->target[0];
	Point *result = (Point *)ctx->target[1];
	Map *mymap = gen->map->clone();
	while(true){
		// pick one object for generating
		int obj = ctx->fetch_one();
		if(obj<0){
			break;
		}
//		log("%d",obj);
		vector<Point *> trace = gen->get_trace(mymap);
		// copy to target
		for(int i=0;i<gen->config.duration;i++){
			result[i*gen->config.num_objects+obj] = *trace[i];
			delete trace[i];
		}
		trace.clear();
	}
	delete mymap;
	return NULL;
}


Point *trace_generator::generate_trace(){
	struct timeval start = get_cur_time();
	Point *ret = (Point *)malloc(config.duration*config.num_objects*sizeof(Point));
	pthread_t threads[config.num_threads];
	query_context tctx;
	tctx.config = config;
	tctx.target[0] = (void *)this;
	tctx.target[1] = (void *)ret;
	tctx.counter = config.num_objects;
	for(int i=0;i<config.num_threads;i++){
		pthread_create(&threads[i], NULL, gentrace, (void *)&tctx);
	}
	for(int i = 0; i < config.num_threads; i++ ){
		void *status;
		pthread_join(threads[i], &status);
	}
	logt("generate traces",start);
	return ret;
}
