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
	ZoneStats total;
	while (std::getline(file, str)&&--limit>0){
		Trip *t = new Trip(str);
		if(map->getMBR()->contain(t->start.coordinate)&&map->getMBR()->contain(t->end.coordinate)){
			int loc = getgrid(&t->end.coordinate);
			zones[loc].count++;
			zones[loc].duration += t->duration();
			double dist = t->start.coordinate.distance(t->end.coordinate, true);
			zones[loc].length += dist;
			total.count++;
			total.duration += t->duration();
			total.length += dist;
		}
		delete t;
	}
	file.close();
	for(int i=dimy-1;i>=0;i--){
		for(int j=0;j<dimx;j++){
			if(zones[i*dimx+j].duration==0||zones[i*dimx+j].length==0){
				zones[i*dimx+j].count = 1;
				zones[i*dimx+j].duration = total.duration/total.count;
				zones[i*dimx+j].length = total.length/total.count;
			}
			assert(zones[i*dimx+j].length/zones[i*dimx+j].duration>0);
			//printf("%.3f\t",zones[i*dimx+j].length*1000.0/zones[i*dimx+j].duration);
		}
		//printf("\n");
	}
}


void trace_generator::rasterize(int num_grids){
	double multi = abs((map->getMBR()->high[1]-map->getMBR()->low[1])/(map->getMBR()->high[0]-map->getMBR()->low[0]));
	step = (map->getMBR()->high[0]-map->getMBR()->low[0])/std::pow(num_grids*1.0/multi,0.5);
	dimx = (map->getMBR()->high[0]-map->getMBR()->low[0])/step+1;
	dimy = (map->getMBR()->high[1]-map->getMBR()->low[1])/step+1;
	zones.resize(dimx*dimy);
}

int trace_generator::getgrid(Point *p){
	assert(step>0);
	int offsety = (p->y-map->getMBR()->low[1])/step;
	int offsetx = (p->x-map->getMBR()->low[0])/step;
	return dimx*offsety+offsetx;
}



Point *trace_generator::get_next(Point *original){
	double xoff = get_rand_double();
	double yoff = get_rand_double();
	double xval = map->getMBR()->low[0]+xoff*(map->getMBR()->high[0]-map->getMBR()->low[0]);
	double yval = map->getMBR()->low[1]+yoff*(map->getMBR()->high[1]-map->getMBR()->low[1]);

	Point *dest = new Point(xval,yval);

	if(!original){

	}else{

	}

	return dest;
}


vector<Point *> trace_generator::get_trace(Map *mymap){
	vector<Point *> ret;
	Point *origin = get_next();
	Point *dest = get_next(origin);
	while(ret.size()<duration){
		mymap->navigate(ret, origin, dest, zones[getgrid(origin)].get_speed());
		// move to another
		delete origin;
		origin = dest;
		dest = get_next(origin);
	}
	for(int i=duration;i<ret.size();i++){
		delete ret[i];
	}
	ret.erase(ret.begin()+duration,ret.end());
	assert(ret.size()==duration);

	delete origin;
	delete dest;
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


