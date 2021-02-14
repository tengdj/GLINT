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
		double portion = md*1.0/duration();
		end.coordinate.x = (end.coordinate.x-start.coordinate.x)*portion+start.coordinate.x;
		end.coordinate.y = (end.coordinate.y-start.coordinate.y)*portion+start.coordinate.y;
		end.timestamp = start.timestamp + md;
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

				if(zones[zid]->target_count.find(ezid)==zones[zid]->target_count.end()){
					zones[zid]->target_count[ezid] = 1;
				}else{
					zones[zid]->target_count[ezid]++;
				}
			}
		}
		delete t;
	}
	file.close();

	// process the empty ones
	if(true){
		int num_not_assigned = 0;
		int update_round = 1;
		do{
			num_not_assigned = 0;
			for(int y=0;y<grid->dimy;y++){
				for(int x=0;x<grid->dimx;x++){
					int zid = y*grid->dimx+x;
					assert(zid<grid->get_grid_num());
					// if no statistics can be collected for this zone
					// simply gather the average statistics of its neighbors
					if(zones[zid]->count==0){
						long nb_count = 0;
						zones[zid]->duration = 0;
						zones[zid]->length = 0;

						for(int yshift=-1;yshift<=1;yshift++){
							for(int xshift=-1;xshift<=1;xshift++){
								int cur_zid = (y+yshift)*grid->dimx+x+xshift;
								if(cur_zid!=zid&&cur_zid>=0&&cur_zid<grid->get_grid_num()){
									nb_count++;
									if(zones[cur_zid]->count>0&&
									   zones[cur_zid]->updated_round<update_round){
										zones[zid]->count += zones[cur_zid]->count;
										zones[zid]->duration += zones[cur_zid]->duration;
										zones[zid]->length += zones[cur_zid]->length;
									}
								}
							}
						}

						if(zones[zid]->count!=0){
							nb_count = min(zones[zid]->count, nb_count);
							zones[zid]->updated_round = update_round;
							//assert(zones[zid]->count >= nb_count);
							zones[zid]->count /= nb_count;
							zones[zid]->duration /= nb_count;
							zones[zid]->length /= nb_count;
							assert(zones[zid]->length>0);
							assert(zones[zid]->duration>0);
							total->count += zones[zid]->count;
							total->duration += zones[zid]->duration;
							total->length += zones[zid]->length;
						}else{
							num_not_assigned++;
						}
					}
				}
			}
			update_round++;
		}while(num_not_assigned>0);
	}

	// reorganize
	sort(ordered_zones.begin(),ordered_zones.end(),orderzone);
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
		// get a start location
		do{
			next->start.timestamp = 0;
			int start_x = -1;
			int start_y = -1;
			// certain portion follows the distribution
			// of analyzed dataset, the rest randomly generate
			if(tryluck(0.2)){
				double target = get_rand_double();
				//log("%f",target);
				double cum = 0;
				for(ZoneStats *z:ordered_zones){
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
		double speed = total->get_speed();
		if(zones[locstart]->count>0){
			speed = zones[locstart]->get_speed();
		}
		// randomly set the speed to [50%, 150%]
		speed *= (1.5-get_rand_double());
		next->end.timestamp = next->start.timestamp+next->length()/speed;
	}
	assert(next->end.timestamp>=next->start.timestamp);
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
	trip->resize(config.duration);
	Point *first_point = new Point(trip->start.coordinate.x,trip->start.coordinate.y);
	ret.push_back(first_point);
	while(ret.size()<config.duration){
		// stay here
		if(trip->start.coordinate.equals(trip->end.coordinate)){
			for(int i=0;i<trip->duration()&&ret.size()<config.duration;i++){
				ret.push_back(new Point(&trip->start.coordinate));
			}
		}else{
			mymap->navigate(ret, &trip->start.coordinate, &trip->end.coordinate, trip->speed());
		}
		// move to another trip following last trip
		Trip *newtrip = next_trip(trip);
		delete trip;
		trip = newtrip;
		trip->resize(config.duration-ret.size());
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
		size_t start = 0;
		size_t end = 0;
		if(!ctx->next_batch(start,end)){
			break;
		}
		for(int obj=start;obj<end;obj++){
			//log("%d",obj);
			vector<Point *> trace = gen->get_trace(mymap);
			// copy to target
			for(int i=0;i<gen->config.duration;i++){
				result[i*gen->config.num_objects+obj] = *trace[i];
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
	Point *ret = (Point *)malloc(config.duration*config.num_objects*sizeof(Point));
	pthread_t threads[config.num_threads];
	query_context tctx;
	tctx.config = config;
	tctx.target[0] = (void *)this;
	tctx.target[1] = (void *)ret;
	tctx.num_objects = config.num_objects;
	tctx.report_gap = 1;
	tctx.batch_size = 100;
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
