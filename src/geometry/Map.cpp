/*
 * Map.cpp
 *
 *  Created on: Jan 11, 2021
 *      Author: teng
 */

#include "Map.h"
#include <time.h>
#include <sys/time.h>
#include <bits/stdc++.h>



Point *Street::close(Street *seg) {
	if(seg==NULL) {
		return NULL;
	}
	if(seg->start==start||seg->start==end) {
		return seg->start;
	}
	if(seg->end==end||seg->end==start) {
		return seg->end;
	}
	return NULL;
}

//whether the target segment interact with this one
//if so, put it in the connected map
bool Street::touch(Street *seg) {
	//if those two streets are connected, record the connection relationship
	//since one of the two streets is firstly added, it is for sure it is unique in others list
	if(close(seg)!=NULL) {
		connected.push_back(seg);
		seg->connected.push_back(this);
		return true;
	}
	return false;
}



/*
 * commit a breadth-first search start from this
 *
 * */
Street *Street::breadthFirst(Street *target, int thread_id) {

	queue<Street *> q;
	q.push(this);
	Street *dest = NULL;
	while(!q.empty()) {
		dest = q.front();
		q.pop();
		if(dest == target) {//found
			break;
		}
		for(Street *sc:dest->connected) {
			if(sc==this){
				continue;
			}
			if(sc->father_from_origin[thread_id]==NULL) {
				sc->father_from_origin[thread_id] = dest;
				q.push(sc);
			}
		}
	}
	if(dest == target){
		return dest;
	}else{
		return NULL;
	}
}

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
 * Map member functions
 *
 * */


/*
 * compare each street pair to see if they connect with each other
 * */
void Map::connect_segments(vector<vector<Street *>> connection) {
	log("connecting streets");
	struct timeval start = get_cur_time();

	int total = 0;
	for(vector<Street *> &connects:connection){
		for(int i=0;i<connects.size()-1;i++){
			for(int j=i+1;j<connects.size();j++){
				connects[i]->connected.push_back(connects[j]);
				connects[j]->connected.push_back(connects[i]);
				total += 2;
			}
		}
	}

	logt("touched %d streets %f",start,streets.size(),total*1.0/streets.size());

	// get the maximum connected component graph
	int *checked = new int[streets.size()];
	for(int i=0;i<streets.size();i++) {
		checked[i] = 0;
	}

	vector<Street *> cur_max_graph;
	int connected = 0;
	int cur_list = 0;
	while(connected<streets.size()){
		queue<Street *> q;
		Street *seed = NULL;
		vector<Street *> tmp;
		for(int i=0;i<streets.size();i++) {
			if(!checked[i]){
				seed = streets[i];
			}
		}
		q.push(seed);
		while(!q.empty()) {
			Street *cur = q.front();
			q.pop();
			if(checked[cur->id]==1){
				continue;
			}
			tmp.push_back(cur);
			checked[cur->id] = 1;
			connected++;
			for(Street *sc:cur->connected) {
				if(!checked[sc->id]) {
					q.push(sc);
				}
			}
		}
		if(cur_max_graph.size()<tmp.size()){
			cur_max_graph.clear();
			cur_max_graph.insert(cur_max_graph.begin(), tmp.begin(), tmp.end());
		}
		tmp.clear();
	}

	// update the street list with only the connected streets
	unordered_set<Street *> included;
	for(Street *st:cur_max_graph){
		included.insert(st);
	}
	for(Street *st:streets){
		if(included.find(st)==included.end()){
			delete st;
		}
	}
	streets.clear();
	streets.insert(streets.end(), cur_max_graph.begin(), cur_max_graph.end());
	for(unsigned int i=0;i<streets.size();i++){
		streets[i]->id = i;
	}
	delete checked;
	logt("connected %d streets",start,streets.size());
}

void Map::dumpTo(const char *path) {

	ofstream wf(path, ios::out | ios::binary|ios::trunc);

	unsigned int num = nodes.size();
	wf.write((char *)&num, sizeof(num));
	for(Point *p:nodes){
		wf.write((char *)&p->x, sizeof(p->x));
		wf.write((char *)&p->y, sizeof(p->y));
	}
	num = streets.size();
	wf.write((char *)&num, sizeof(num));
	for(Street *s:streets){
		wf.write((char *)&s->start->id, sizeof(s->start->id));
		wf.write((char *)&s->end->id, sizeof(s->end->id));
		num = s->connected.size();
		wf.write((char *)&num, sizeof(num));
		for(Street *cs:s->connected){
			wf.write((char *)&cs->id, sizeof(cs->id));
		}
	}
	wf.close();
	log("dumped to %s",path);
}


void Map::loadFromCSV(const char *path){

	std::ifstream file(path);
	std::string str;
	vector<string> fields;
	//skip the head
	std::getline(file, str);
	map<string, Point *> nodeset;
	char cotmp[256];
	vector<double> values;
	unsigned int id = 0;
	Point *start_point = NULL;
	Point *end_point = NULL;
	vector<vector<Street *>> connections;
	while (std::getline(file, str)){
		// Process str
		 fields.clear();
		 tokenize(str, fields);
		 if(fields.size()<=2){
			 continue;
		 }
		 string geo = fields[1];
		 if(geo.size()<18){
			 continue;
		 }
		 values = parse_double_values(geo);
		 if(values.size()==0){
			 continue;
		 }
		 assert(values.size()%2==0);

		 double start[2] = {values[0], values[1]};
		 double end[2] = {values[values.size()-2], values[values.size()-1]};

		 sprintf(cotmp, "%.14f_%.14f", values[0], values[1]);

		 if(nodeset.find(cotmp)==nodeset.end()){
			 Point *p = new Point(start[0],start[1]);
			 p->id = nodeset.size();
			 nodes.push_back(p);
			 vector<Street *> cs;
			 connections.push_back(cs);
			 nodeset[cotmp] = p;
		 }
		 start_point = nodeset[cotmp];
		 sprintf(cotmp, "%.14f_%.14f", values[values.size()-2], values[values.size()-1]);
		 if(nodeset.find(cotmp)==nodeset.end()){
			 Point *p = new Point(end[0],end[1]);
			 p->id = nodeset.size();
			 nodes.push_back(p);
			 vector<Street *> cs;
			 connections.push_back(cs);
			 nodeset[cotmp] = p;
		 }
		 end_point = nodeset[cotmp];
		 Street *ns = new Street(id++,start_point,end_point);
		 streets.push_back(ns);
		 connections[start_point->id].push_back(ns);
		 connections[end_point->id].push_back(ns);
	}
	nodeset.clear();
	connect_segments(connections);
	for(vector<Street *> &cs:connections){
		cs.clear();
	}
	connections.clear();
	log("%ld nodes and %ld streets are loaded",nodes.size(),streets.size());
	getMBR();
}


void Map::loadFrom(const char *path) {

	struct timeval start_time = get_cur_time();
	ifstream in(path, ios::in | ios::binary);

	vector<vector<unsigned int>> connected;

	unsigned int num;
	in.read((char *)&num, sizeof(num));
	nodes.resize(num);
	for(unsigned int i=0;i<num;i++){
		Point *p = new Point();
		p->id = i;
		in.read((char *)&p->x, sizeof(p->x));
		in.read((char *)&p->y, sizeof(p->y));
		nodes[i] = p;
	}
	in.read((char *)&num, sizeof(num));
	streets.resize(num);
	connected.resize(num);
	int slen = num;
	unsigned int id = 0;
	Point *start = NULL;
	Point *end = NULL;
	for(;id<slen;id++){
		in.read((char *)&num, sizeof(num));
		start = nodes[num];
		in.read((char *)&num, sizeof(num));
		end = nodes[num];
		in.read((char *)&num, sizeof(num));
		int len = num;
		vector<unsigned int> cons;
		for(int j=0;j<len;j++){
			in.read((char *)&num, sizeof(num));
			cons.push_back(num);
		}
		connected[id] = cons;
		assert(start&&end);
		streets[id] = new Street(id, start, end);
	}

	for(int i=0;i<slen;i++){
		for(unsigned int sl:connected[i]){
			streets[i]->connected.push_back(streets[sl]);
		}
		connected[i].clear();
	}
	connected.clear();
	in.close();
	getMBR();
	logt("loaded %d nodes %d streets from %s",start_time,nodes.size(), streets.size(), path);
}

Street *Map::nearest(Point *target){
	assert(target);
	Street *ret;
	double dist = DBL_MAX;
	for(Street *st:streets) {
		double d = st->distance(target);
		if(d<dist){
			ret = st;
			dist = d;
		}
	}
	return ret;
}

/*
 * simulate the trajectory of the trip.
 * with the given streets the trip has covered, generate a list of points
 * that the taxi may appear at a given time
 *
 * */
int Map::navigate(vector<Point *> &positions, Point *origin, Point *dest, double speed, int thread_id){

	assert(origin);
	assert(dest);
	assert(speed>0);
	assert(thread_id>=0);

	// get the closest points streets to the source and destination points
	vector<Street *> ret;
	Street *o = nearest(origin);
	Street *d = nearest(dest);

	//conduct a breadth first query to get a list of streets
	for(Street *s:streets) {
		s->father_from_origin[thread_id] = NULL;
	}
	Street *s = o->breadthFirst(d, thread_id);
	assert(s);
	do{
		ret.push_back(s);
		s = s->father_from_origin[thread_id];
	}while(s!=NULL);
	reverse(ret.begin(),ret.end());

	// convert the street sequence to point sequences
	vector<Point *> trajectory;
	Point *cur;
	if(ret.size()==1){
		cur = ret[0]->start;
	}else{
		cur = ret[0]->close(ret[1]);
		// swap to other end
		if(ret[0]->start==cur){
			cur = ret[0]->end;
		}else{
			cur = ret[0]->start;
		}
	}

	for(int i=0;i<ret.size();i++){
		trajectory.push_back(cur);
		if(ret[i]->start==cur){
			cur = ret[i]->end;
		}else{
			cur = ret[i]->start;
		}
	}
	trajectory.push_back(cur);

	// quantify the street sequence to generate a list of
	// points with fixed gap
	double dist_from_origin = 0;
	int inserted = 0;
	for(int i=0;i<trajectory.size()-1;i++) {
		Point *cur_start = trajectory[i];
		Point *cur_end = trajectory[i+1];
		double length = cur_start->distance(*cur_end, true);
		double next_dist_from_origin = dist_from_origin+=length;
		double cur_dis = ((int)(next_dist_from_origin/speed)+1)*speed-dist_from_origin;
		while(cur_dis<length) {//have other position can be reported in this street
			double cur_portion = cur_dis/length;
			//now get the longitude and latitude and timestamp for current event and add to return list
			Point *p = new Point(cur_start->x+(cur_end->x-cur_start->x)*cur_portion,
								 cur_start->y+(cur_end->y-cur_start->y)*cur_portion);
			positions.push_back(p);
			cur_dis += speed;
			inserted++;
		}

		//move to next street
		dist_from_origin = next_dist_from_origin;
	}

	trajectory.clear();
	ret.clear();
	return inserted;
}




/*
 *
 * functions for generating simulated traces of an object
 * based on the real world statistics
 *
 * */


void Map::print_region(box region){
	printf("MULTILINESTRING(");
	bool first = true;
	for(int i=0;i<streets.size();i++){
		if(region.contain(*streets[i]->start)||region.contain(*streets[i]->end)){
			if(!first){
				printf(",");
			}else{
				first = false;
			}
			printf("(%f %f, %f %f)",streets[i]->start->x,streets[i]->start->y,streets[i]->end->x,streets[i]->end->y);
		}
	}
	printf(")\n");
}



void Map::analyze_trips(const char *path, int limit){
	std::ifstream file(path);
	std::string str;
	//skip the head
	std::getline(file, str);
	getMBR();
	ZoneStats total;
	while (std::getline(file, str)&&--limit>0){
		Trip *t = new Trip(str);
		if(mbr->contain(t->start.coordinate)&&mbr->contain(t->end.coordinate)){
			int loc = this->getgrid(&t->end.coordinate);
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


void Map::rasterize(int num_grids){
	assert(mbr);
	double multi = abs((mbr->high[1]-mbr->low[1])/(mbr->high[0]-mbr->low[0]));
	step = (mbr->high[0]-mbr->low[0])/std::pow(num_grids*1.0/multi,0.5);
	dimx = (mbr->high[0]-mbr->low[0])/step+1;
	dimy = (mbr->high[1]-mbr->low[1])/step+1;
	zones.resize(dimx*dimy);
}

int Map::getgrid(Point *p){
	assert(step>0);
	int offsety = (p->y-mbr->low[1])/step;
	int offsetx = (p->x-mbr->low[0])/step;
	return dimx*offsety+offsetx;
}



Point *Map::get_next(Point *original){
	double xoff = get_rand_double();
	double yoff = get_rand_double();
	double xval = mbr->low[0]+xoff*(mbr->high[0]-mbr->low[0]);
	double yval = mbr->low[1]+yoff*(mbr->high[1]-mbr->low[1]);

	Point *dest = new Point(xval,yval);

	if(!original){

	}else{

	}

	return dest;
}


vector<Point *> Map::get_trace(int thread_id, int duration){
	vector<Point *> ret;
	Point *origin = get_next();
	Point *dest = get_next(origin);
	while(ret.size()<duration){
		//origin->print();
		//dest->print();
		//printf("%f\n",zones[getgrid(origin)].get_speed());
		navigate(ret, origin, dest, zones[getgrid(origin)].get_speed(), thread_id);
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
	int thread_id = 0;
	int duration = 0;
	int *counter;
	int max_num = 0;
	Map *mp = NULL;
	double *result = NULL;
};


void *gentrace(void *arg){
	trace_context *ctx = (trace_context *)arg;
	log("thread %d started",ctx->thread_id);

	while(true){
		int cur_t = 0;
		lock();
		cur_t = (*ctx->counter)++;
		unlock();
		if(cur_t>=ctx->max_num){
			break;
		}
		//log("%d",cur_t);
		vector<Point *> trace = ctx->mp->get_trace(ctx->thread_id, ctx->duration);
		double *points = ctx->result+2*cur_t*ctx->duration;
		for(Point *p:trace){
			*points++ = p->x;
			*points++ = p->y;
			delete p;
		}
		trace.clear();
	}

	return NULL;
}


double *Map::generate_trace(int duration, int count,int num_threads){
	double *ret = new double[duration*count*2];
	if(num_threads<=0){
		num_threads = get_num_threads();
	}
	pthread_t threads[num_threads];
	trace_context ctx[num_threads];
	int counter = 0;
	for(int i=0;i<num_threads;i++){
		ctx[i].thread_id = i;
		ctx[i].mp = this;
		ctx[i].counter = &counter;
		ctx[i].max_num = count;
		ctx[i].result = ret;
		ctx[i].duration = duration;
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

