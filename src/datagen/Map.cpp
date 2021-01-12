/*
 * Map.cpp
 *
 *  Created on: Jan 11, 2021
 *      Author: teng
 */

#include "Map.h"
#include <time.h>
#include <sys/time.h>


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
	printf("connecting streets\n");
	for(int i=0;i<streets.size()-1;i++) {
		for(int j=i+1;j<streets.size();j++) {
			streets[i]->touch(streets[j]);
		}
	}
}

void Map::dumpTo(const char *path) {

	printf("dumping to %s\n",path);
	ofstream wf(path, ios::out | ios::binary);

	unsigned int num = nodes.size();
	wf.write((char *)&num, sizeof(num));
	for(Point *p:nodes){
		wf.write((char *)&p->x, sizeof(p->x));
		wf.write((char *)&p->y, sizeof(p->y));
	}
	num = streets.size();
	wf.write((char *)&num, sizeof(num));
	for(Street *s:streets){
		wf.write((char *)&s->id, sizeof(s->id));
		wf.write((char *)&s->start->id, sizeof(s->start->id));
		wf.write((char *)&s->end->id, sizeof(s->end->id));
		num = s->connected.size();
		wf.write((char *)&num, sizeof(num));
		for(Street *cs:s->connected){
			wf.write((char *)&cs->id, sizeof(cs->id));
		}

	}
}

void Map::loadFrom(const char *path) {
	printf("loading from %s\n",path);

	ifstream in(path, ios::in | ios::binary);

	vector<vector<unsigned int>> connected;

	unsigned int num;
	in.read((char *)&num, sizeof(num));
	nodes.resize(num);
	for(int i=0;i<num;i++){
		Point *p = new Point();
		in.read((char *)&p->id, sizeof(p->id));
		in.read((char *)&p->x, sizeof(p->x));
		in.read((char *)&p->y, sizeof(p->y));
		nodes[p->id] = p;
	}
	in.read((char *)&num, sizeof(num));
	streets.resize(num);
	connected.resize(num);
	int slen = num;
	for(int i=0;i<slen;i++){
		Street *s = new Street();

		in.read((char *)&num, sizeof(num));
		s->id = num;
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
		connected[s->id] = cons;
		streets[s->id] = s;
	}

	for(int i=0;i<slen;i++){
		for(unsigned int sl:connected[i]){
			streets[i]->connected.push_back(streets[sl]);
		}
		connected[i].clear();
	}
	connected.clear();
}

vector<Street *> Map::nearest(Point *target, int limit){
	vector<Street *> ret;
	vector<double> dist;
	for(Street *st:streets) {
		double d = distance_point_to_segment(target, st);
		if(dist.size()==0) {
			dist.push_back(d);
			ret.push_back(st);
			continue;
		}
		//the queue is full and the distance is bigger than or equal to the current minimum
		if(dist.size()>=limit&&d>=dist[dist.size()-1]) {
			continue;
		}
		//otherwise, insert current street into the return list, evict the tail
		for(int insert_into = 0;insert_into<dist.size();insert_into++) {
			if(dist[insert_into]>=d) {
				ret.insert(ret.begin()+insert_into, st);
				dist.insert(dist.begin()+insert_into, d);
				break;
			}
		}

		if(ret.size()>limit) {
			ret.pop_back();
			dist.pop_back();
		}
	}
	dist.clear();
	return ret;
}

vector<Street *> Map::navigate(Point *origin, Point *dest){

	vector<Street *> ret;
	vector<Street *> originset = nearest(origin, 5);
	vector<Street *> destset = nearest(dest, 5);

	for(Street *o:originset) {
		for(Street *d:destset) {
			//initialize
			for(Street *s:streets) {
				s->father_from_origin = NULL;
			}
			Street *s = o->breadthFirst(d->id);
			if(s!=NULL) {
				while(s->father_from_origin!=NULL) {
					ret.push_back(s);
					s = s->father_from_origin;
				}
				break;
			}
		}

		if(ret.size()>0) {
			break;
		}
	}

	vector<Street *> reversed;
	reversed.resize(ret.size());
	for(int i=ret.size()-1;i>=0;i--) {
		reversed[ret.size()-1-i] = ret[i];
	}
	ret.clear();
	return reversed;

}

Trip::Trip(string cols[]){

	start_time = 0;
	char tmp[2];
	tmp[0] = cols[2][11];
	tmp[1] = cols[2][12];
	start_time += atoi(tmp)*3600;
	tmp[0] = cols[2][14];
	tmp[1] = cols[2][15];
	start_time += atoi(tmp)*60;
	tmp[0] = cols[2][17];
	tmp[1] = cols[2][18];
	start_time += atoi(tmp);

	end_time = 0;
	tmp[0] = cols[3][11];
	tmp[1] = cols[3][12];
	end_time += atoi(tmp)*3600;
	tmp[0] = cols[3][14];
	tmp[1] = cols[3][15];
	end_time += atoi(tmp)*60;
	tmp[0] = cols[3][17];
	tmp[1] = cols[3][18];
	end_time += atoi(tmp);


	start_location = new Point(atof(cols[18].c_str()),atof(cols[17].c_str()));
	end_location = new Point(atof(cols[21].c_str()),atof(cols[20].c_str()));
}
/*
 * simulate the trajectory of the trip.
 * with the given streets the trip has covered, generate a list of points
 * that the taxi may appear at a given time
 *
 * */
vector<Event *> Trip::getCurLocations(vector<Street *> st) {

	int duration_time = end_time-start_time;

	vector<Event *> positions;

	double total_length = 0;
	for(Street *s:st) {
		total_length += s->getLength();
	}
	//ever second
	double step = total_length/duration_time;
	Point *origin = NULL;
	double dist_from_origin = 0;
	if(st.size()==1) {
		origin = st[0]->start;
	}else {
		if(st[0]->start->equals(st[1]->start)||st[0]->start->equals(st[1]->end)) {
			origin = st[0]->end;
		}else {
			origin = st[0]->start;
		}
	}

	for(Street *s:st) {
		bool from_start = origin->equals(s->start);
		double next_dist_from_origin = dist_from_origin+=s->length;

		Point *cur_start = from_start?s->start:s->end;
		Point *cur_end = from_start?s->end:s->start;

		double cur_dis = ((int)(next_dist_from_origin/step)+1)*step-dist_from_origin;
		while(cur_dis<s->length) {//have other position can be reported in this street
			double cur_portion = cur_dis/s->length;
			//now get the longitude and latitude and timestamp for current event and add to return list
			Event *cp = new Event();
			cp->timestamp = (long)(((cur_dis+dist_from_origin)*1000/total_length)*duration_time+this->start_time);
			cp->coordinate = new Point(cur_start->x+(cur_end->x-cur_start->x)*cur_portion,
					cur_start->y+(cur_end->y-cur_start->y)*cur_portion);
			positions.push_back(cp);
			cur_dis += step;
		}


		//now cut cur_start->cur_end according to the start and step

		//move to next street
		dist_from_origin = next_dist_from_origin;
		if(from_start) {
			origin = s->end;
		}else {
			origin = s->start;
		}
	}

	return positions;

}
