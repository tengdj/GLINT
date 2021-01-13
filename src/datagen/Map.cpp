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

void Map::clear() {
	for(Street *s:streets) {
		delete s;
	}
	streets.clear();
	for(Point *p:nodes){
		delete p;
	}
	nodes.clear();
}



/*
 * compare each street pair to see if they connect with each other
 * */
void Map::connect_segments() {
	log("connecting streets");
	struct timeval start = get_cur_time();

	int total = 0;
	for(Point *p:nodes){
		for(int i=0;i<p->connects.size()-1;i++){
			for(int j=i+1;j<p->connects.size();j++){
				p->connects[i]->connected.push_back(p->connects[j]);
				p->connects[j]->connected.push_back(p->connects[i]);
				total += 2;
			}
		}
	}

	logt("touched %d streets %f",start,streets.size(),total*1.0/streets.size());
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

		 Street *st = new Street();
		 st->id = streets.size();
		 sprintf(cotmp, "%.14f_%.14f", values[0], values[1]);

		 if(nodeset.find(cotmp)==nodeset.end()){
			 Point *p = new Point(start[0],start[1]);
			 p->id = nodeset.size();
			 nodes.push_back(p);
			 nodeset[cotmp] = p;
		 }
		 st->start = nodeset[cotmp];
		 sprintf(cotmp, "%.14f_%.14f", values[values.size()-2], values[values.size()-1]);
		 if(nodeset.find(cotmp)==nodeset.end()){
			 Point *p = new Point(end[0],end[1]);
			 p->id = nodeset.size();
			 nodes.push_back(p);
			 nodeset[cotmp] = p;
		 }
		 st->end = nodeset[cotmp];
		 streets.push_back(st);
		 st->start->connects.push_back(st);
		 st->end->connects.push_back(st);
	}
	nodeset.clear();
	log("%ld nodes and %ld streets are loaded",nodes.size(),streets.size());

	this->connect_segments();
}


void Map::loadFrom(const char *path) {

	struct timeval start = get_cur_time();
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
	for(unsigned int i=0;i<slen;i++){
		Street *s = new Street();
		s->id = i;
		in.read((char *)&num, sizeof(num));
		s->start = nodes[num];
		in.read((char *)&num, sizeof(num));
		s->end = nodes[num];
		in.read((char *)&num, sizeof(num));
		int len = num;
		vector<unsigned int> cons;
		for(int j=0;j<len;j++){
			in.read((char *)&num, sizeof(num));
			cons.push_back(num);
		}
		connected[i] = cons;
		streets[i] = s;
	}

	for(int i=0;i<slen;i++){
		for(unsigned int sl:connected[i]){
			streets[i]->connected.push_back(streets[sl]);
		}
		connected[i].clear();
	}
	connected.clear();
	in.close();
	logt("loaded %d nodes %d streets from %s",start,nodes.size(), streets.size(), path);
}

Street *Map::nearest(Point *target){
	Street *ret;
	double dist = DBL_MAX;
	for(Street *st:streets) {
		double d = distance_point_to_segment(target, st);
		if(d<dist){
			ret = st;
			dist = d;
		}
	}
	return ret;
}

vector<Point *> Map::navigate(Point *origin, Point *dest){

	vector<Street *> ret;
	Street *o = nearest(origin);
	Street *d = nearest(dest);

//	printf("%d LINESTRING(%f %f, %f %f)\n",o->id, o->start->x,o->start->y,o->end->x,o->end->y);
//	printf("%d LINESTRING(%f %f, %f %f)\n",d->id, d->start->x,d->start->y,d->end->x,d->end->y);

	//initialize
	for(Street *s:streets) {
		s->father_from_origin = NULL;
	}
	Street *s = o->breadthFirst(d->id);
	assert(s);
	do{
		ret.push_back(s);
		s = s->father_from_origin;
	}while(s!=NULL);

	vector<Street *> reversed;
	reversed.resize(ret.size());
	for(int i=ret.size()-1;i>=0;i--) {
		reversed[ret.size()-1-i] = ret[i];
	}
	ret.clear();

	vector<Point *> trajectory;
	if(reversed.size()==1){
		trajectory.push_back(reversed[0]->start);
		trajectory.push_back(reversed[0]->end);
		return trajectory;
	}

	Point *cur = reversed[0]->close(reversed[1]);
	if(reversed[0]->start==cur){
		cur = reversed[0]->end;
	}else{
		cur = reversed[0]->start;
	}

	for(int i=0;i<reversed.size();i++){
		trajectory.push_back(cur);
		if(reversed[i]->start==cur){
			cur = reversed[i]->end;
		}else{
			cur = reversed[i]->start;
		}
	}
	trajectory.push_back(cur);

	return trajectory;

}

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
