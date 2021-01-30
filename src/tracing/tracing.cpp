/*
 * generator.cpp
 *
 *  Created on: Jan 19, 2021
 *      Author: teng
 */

#include "tracing.h"
#include "../index/QTree.h"


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
 * member functions for grid class
 *
 * */

void Grid::rasterize(int num_grids){
	double multi = abs((space.high[1]-space.low[1])/(space.high[0]-space.low[0]));
	step = (space.high[0]-space.low[0])/std::pow(num_grids*1.0/multi,0.5);
	while(true){
		dimx = (space.high[0]-space.low[0])/step+1;
		dimy = (space.high[1]-space.low[1])/step+1;
		// increase the step if too many grids are generated
		if(dimx*dimy>num_grids){
			step = step*1.01;
		}else{
			break;
		}
	}
}

int Grid::getgrid(double x, double y){
	assert(step>0);
	int offsety = (y-space.low[1])/step;
	int offsetx = (x-space.low[0])/step;
	int gid = dimx*offsety+offsetx;
	assert(gid<=dimx*dimy && gid>=0);
	return gid;
}

int Grid::getgrid(Point *p){
	return getgrid(p->x, p->y);
}

Point Grid::get_random_point(int xoff, int yoff){
	double xrand = get_rand_double();
	double yrand = get_rand_double();
	double xval,yval;
	if(xoff==-1||yoff==-1){
		xval = space.low[0]+xrand*(space.high[0]-space.low[0]);
		yval = space.low[1]+yrand*(space.high[1]-space.low[1]);
	}else{
		xval = space.low[0]+xoff*step+xrand*step;
		yval = space.low[1]+yoff*step+yrand*step;
	}
	return Point(xval, yval);
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
			int zid = grid->getgrid(&t->start.coordinate);
			zones[zid]->count++;
			zones[zid]->duration += t->duration();
			double dist = t->start.coordinate.distance(t->end.coordinate, true);
			zones[zid]->length += dist;
			total->count++;
			total->duration += t->duration();
			total->length += dist;

			int ezid = grid->getgrid(&t->end.coordinate);
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
	int locstart = grid->getgrid(&next->start.coordinate);
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
		int gid = grid->getgrid(&next->end.coordinate);
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
		int obj = 0;
		lock();
		obj = --gen->counter;
		unlock();
		if(obj<0){
			break;
		}
		//log("%d",obj);
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
	query_context tctx[config.num_threads];
	for(int i=0;i<config.num_threads;i++){
		tctx[i].config = config;
		tctx[i].target[0] = (void *)this;
		tctx[i].target[1] = (void *)ret;
	}
	for(int i=0;i<config.num_threads;i++){
		pthread_create(&threads[i], NULL, gentrace, (void *)&tctx[i]);
	}
	for(int i = 0; i < config.num_threads; i++ ){
		void *status;
		pthread_join(threads[i], &status);
	}
	logt("generate traces",start);
	return ret;
}

/*
 * functions for tracer
 *
 * */

bool myfunction (QTNode *n1, QTNode *n2) {
	return n1->objects.size()<n2->objects.size();
}

void tracer::process_qtree(){
	struct timeval start = get_cur_time();
	QConfig qconfig;
	qconfig.reach_distance = config.reach_distance;
	qconfig.max_objects = config.max_objects_per_grid;
	qconfig.x_buffer = config.reach_distance*degree_per_kilometer_longitude(mbr.low[1])/1000;
	qconfig.y_buffer = config.reach_distance*degree_per_kilometer_latitude/1000;
	//printf("%f %f %f %f\n",qconfig.x_buffer, qconfig.y_buffer, qconfig.x_buffer/degree_per_kilometer_longitude(mbr.low[1]),qconfig.y_buffer/degree_per_kilometer_latitude);
	QTNode *qtree = new QTNode(mbr);
	qtree->set_config(&qconfig);

	for(int o=0;o<config.num_objects;o++){
		Point *p = trace+o;
		assert(mbr.contain(*p));
		qtree->insert(p);
	}

	logt("building qtree with %ld points with %d max_objects %d leafs", start, qconfig.num_objects, qconfig.max_objects, qconfig.num_leafs);
	qtree->fix_structure();

	// test contact tracing
	vector<QTNode *> nodes;
	size_t counter = 0;
	size_t reached = 0;
	vector<QTNode *> grids;
	grids.resize(config.num_objects);
	for(int t=0;t<config.duration;t++){
		for(int o=0;o<config.num_objects;o++){
			Point *p = trace+t*config.num_objects+o;
			qtree->insert(p);
		}
		qtree->get_leafs(nodes,false);
		sort(nodes.begin(),nodes.end(),myfunction);
		int tt = 0;
		int griddiff = 0;
		for(QTNode *n:nodes){
			int len = n->objects.size();
			for(int i=0;i<len;i++){
				int oid = (n->objects[i]-trace-t*config.num_objects);
				griddiff += (grids[oid]!=n);
				grids[oid] = n;
			}
			//n->print_node();
			if(len>2){
				for(int i=0;i<len-1;i++){
					for(int j=i+1;j<len;j++){
						double dist = n->objects[i]->distance(*n->objects[j], true)*1000;
						//log("%f",dist);
						if(dist<config.reach_distance){
							reached++;
						}
						counter++;
					}
				}
			}
		}
		log("%d %d",t,griddiff);
		nodes.clear();
		qtree->fix_structure();
	}
	delete qtree;
	logt("contact trace with %ld calculation use QTree %ld connected",start,counter,reached);
}

void tracer::process_fixgrid(){
	struct timeval start = get_cur_time();
	// test contact tracing
	int counter = 0;
	vector<vector<Point *>> grids;
	Grid grid(mbr, config.num_grids);
	log("%f",grid.get_step()*1000);
	grids.resize(grid.get_grid_num()+1);
	vector<int> formergrid;
	formergrid.resize(config.num_objects);
	vector<int> gridcount;
	gridcount.resize(grid.get_grid_num()+1);
	for(int t=0;t<config.duration;t++){
		int diff = 0;
		for(int o=0;o<config.num_objects;o++){
			Point *p = trace+t*config.num_objects+o;
			int gid = grid.getgrid(p);
			if(gid!=formergrid[o]){
				diff++;
				formergrid[o] = gid;
			}
			grids[gid].push_back(p);
			gridcount[gid]++;
		}
		sort(gridcount.begin(),gridcount.end(),greater<int>());
		for(int i=0;i<gridcount.size();i++){
			if(!gridcount[i]){
				break;
			}
			cout<<i<<" "<<gridcount[i]<<endl;
		}
		//  cout<<diff<<endl;
		for(vector<Point *> &ps:grids){
			int len = ps.size();
			if(len>=2){
				for(int i=0;i<len-1;i++){
					for(int j=i+1;j<len;j++){
						ps[i]->distance(*ps[j], true);
						counter++;
					}
				}
			}
			ps.clear();
		}
	}
	grids.clear();
	logt("contact trace with %d calculation use fixed grid",start,counter);
}


void tracer::dumpTo(const char *path) {
	struct timeval start_time = get_cur_time();
	ofstream wf(path, ios::out|ios::binary|ios::trunc);
	wf.write((char *)&config.num_objects, sizeof(config.num_objects));
	wf.write((char *)&config.duration, sizeof(config.duration));
	wf.write((char *)&mbr, sizeof(mbr));
	size_t num_points = config.duration*config.num_objects;
	wf.write((char *)trace, sizeof(Point)*num_points);
	wf.close();
	logt("dumped to %s",start_time,path);
}

void tracer::loadFrom(const char *path) {

	int total_num_objects;
	int total_duration;
	struct timeval start_time = get_cur_time();
	ifstream in(path, ios::in | ios::binary);
	in.read((char *)&total_num_objects, sizeof(total_num_objects));
	in.read((char *)&total_duration, sizeof(total_duration));
	in.read((char *)&mbr, sizeof(mbr));
	assert(config.duration<=total_duration);
	assert(config.num_objects<=total_num_objects);

	trace = (Point *)malloc(config.duration*config.num_objects*sizeof(Point));
	for(int i=0;i<config.duration;i++){
		in.read((char *)(trace+i*config.num_objects), config.num_objects*sizeof(Point));
		if(total_num_objects>config.num_objects){
			in.seekg((total_num_objects-config.num_objects)*sizeof(Point), ios_base::cur);
		}
	}

	in.close();
	logt("loaded %d objects last for %d seconds from %s",start_time, config.num_objects, config.duration, path);
	owned_trace = true;
}

void tracer::print_trace(double sample_rate){
	print_points(trace,config.num_objects,sample_rate);
}


