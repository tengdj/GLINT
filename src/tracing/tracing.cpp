/*
 * generator.cpp
 *
 *  Created on: Jan 19, 2021
 *      Author: teng
 */

#include "tracing.h"



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
	std::ifstream file(path);
	std::string str;
	//skip the head
	std::getline(file, str);
	ZoneStats *total = new ZoneStats(0);
	while (std::getline(file, str)&&--limit>0){
		Trip *t = new Trip(str);
		if(map->getMBR()->contain(t->start.coordinate)&&map->getMBR()->contain(t->end.coordinate)){
			int zid = getgrid(&t->end.coordinate);
			zones[zid]->count++;
			zones[zid]->duration += t->duration();
			double dist = t->start.coordinate.distance(t->end.coordinate, true);
			zones[zid]->length += dist;
			total->count++;
			total->duration += t->duration();
			total->length += dist;
		}
		delete t;
	}
	file.close();
	for(int y=0;y<dimy;y++){
		for(int x=0;x<=dimx;x++){
			int zid = y*dimx+x;
			assert(zid<=dimx*dimy);
			if(zones[zid]->duration==0||zones[zid]->length==0){
				zones[zid]->count = 1;
				zones[zid]->duration = total->duration/total->count;
				zones[zid]->length = total->length/total->count;
			}
			assert(zones[zid]->length/zones[zid]->duration>0);
			//printf("%.3f\t",zones[zid]->length*1000.0/zones[zid]->duration);
		}
		//printf("\n");
	}
	delete total;
}


void trace_generator::rasterize(int num_grids){
	double multi = abs((map->getMBR()->high[1]-map->getMBR()->low[1])/(map->getMBR()->high[0]-map->getMBR()->low[0]));
	step = (map->getMBR()->high[0]-map->getMBR()->low[0])/std::pow(num_grids*1.0/multi,0.5);
	while(true){
		dimx = (map->getMBR()->high[0]-map->getMBR()->low[0])/step+1;
		dimy = (map->getMBR()->high[1]-map->getMBR()->low[1])/step+1;
		// increase the step if too many grids are generated
		if(dimx*dimy>num_grids){
			step = step*1.01;
		}else{
			break;
		}
	};
	zones.resize(dimx*dimy+1);
	for(int i=0;i<zones.size();i++){
		zones[i] = new ZoneStats(i);
	}
}

int trace_generator::getgrid(Point *p){
	assert(step>0);
	int offsety = (p->y-map->getMBR()->low[1])/step;
	int offsetx = (p->x-map->getMBR()->low[0])/step;
	int gid = dimx*offsety+offsetx;
	if(gid>dimx*dimy){
		p->print();
		map->getMBR()->print();
		cout<<gid<<" "<<dimx*dimy<<" "<<map->getMBR()->contain(*p)<<endl;;
	}
	assert(gid<=dimx*dimy);
	return gid;
}


/*
 * simulate next move.
 *
 * */

Trip *trace_generator::next_trip(Trip *former){
	Trip *next = new Trip();
	if(former==NULL){
		double xoff = get_rand_double();
		double yoff = get_rand_double();
		double xval = map->getMBR()->low[0]+xoff*(map->getMBR()->high[0]-map->getMBR()->low[0]);
		double yval = map->getMBR()->low[1]+yoff*(map->getMBR()->high[1]-map->getMBR()->low[1]);
		next->start.coordinate = Point(xval,yval);
		next->start.timestamp = 0;
	}else{
		next->start = former->end;
	}

	// now generate the next destination according to current position
	int locstart = getgrid(&next->start.coordinate);
	double rest = get_rand_double()*zones[locstart]->rate_sleep;
	// rest for a while, the time is adjustable
	if(tryluck(rest)){
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
			dest = get_rand_number(dimx*dimy);
		}

		int x = dest%dimx;
		int y = dest/dimx;

		double xoff = get_rand_double();
		double yoff = get_rand_double();
		double xval = map->getMBR()->low[0]+x*step+xoff*step;
		double yval = map->getMBR()->low[1]+y*step+yoff*step;
		next->end.coordinate = Point(xval,yval);
		next->end.timestamp = next->start.timestamp+next->length(true)/zones[locstart]->get_speed();
		int gid = getgrid(&next->end.coordinate);
	}

	return next;
}


vector<Point *> trace_generator::get_trace(Map *mymap){
	vector<Point *> ret;
	Trip *trip = next_trip();
	while(ret.size()<duration){
		// stay here
		if(trip->start.coordinate.equals(trip->start.coordinate)){
			for(int i=0;i<trip->duration()&&ret.size()<duration;i++){
				ret.push_back(new Point(&trip->start.coordinate));
			}
		}else{
			double speed = trip->end.coordinate.distance(trip->start.coordinate, true)/trip->duration();
			mymap->navigate(ret, &trip->start.coordinate, &trip->end.coordinate, speed);
		}
		// move to another
		Trip *newtrip = next_trip(trip);
		delete trip;
		trip = newtrip;
	}
	for(int i=duration;i<ret.size();i++){
		delete ret[i];
	}
	ret.erase(ret.begin()+duration,ret.end());
	assert(ret.size()==duration);

	delete trip;
	return ret;
}


class trace_context{
public:
	trace_generator *gen = NULL;
	double *result = NULL;
};


void *gentrace(void *arg){
	trace_context *ctx = (trace_context *)arg;
	trace_generator *gen = ctx->gen;
	Map *mymap = gen->map->clone();
	while(true){
		// pick one object for generating
		int cur_t = 0;
		lock();
		cur_t = --gen->counter;
		unlock();
		if(cur_t<0){
			break;
		}
		//log("%d",cur_t);
		vector<Point *> trace = ctx->gen->get_trace(mymap);

		// copy to target
		double *points = ctx->result+2*cur_t*gen->duration;
		for(Point *p:trace){
			*points++ = p->x;
			*points++ = p->y;
			delete p;
		}
		trace.clear();
	}
	delete mymap;

	return NULL;
}


double *trace_generator::generate_trace(){
	double *ret = new double[duration*counter*2];
	pthread_t threads[num_threads];
	trace_context ctx[num_threads];
	for(int i=0;i<num_threads;i++){
		ctx[i].gen = this;
		ctx[i].result = ret;
	}
	for(int i=0;i<num_threads;i++){
		pthread_create(&threads[i], NULL, gentrace, (void *)&ctx[i]);
	}
	for(int i = 0; i < num_threads; i++ ){
		void *status;
		pthread_join(threads[i], &status);
	}

	return ret;
}


