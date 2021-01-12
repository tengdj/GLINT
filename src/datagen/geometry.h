/*
 * geometry.h
 *
 *  Created on: Jan 11, 2021
 *      Author: teng
 */

#ifndef DATAGEN_GEOMETRY_H_
#define DATAGEN_GEOMETRY_H_

#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <math.h>
#include <string>

using namespace std;

class Point{
public:
	unsigned int id = 0;
	double x = 0;
	double y = 0;
	Point(double xx, double yy){
		x = xx;
		y = yy;
	}
	Point(){}
	double distance(Point &p){
		return sqrt((p.x-x)*(p.x-x)+(p.y-y)*(p.y-y));
	}
	bool equals(Point *p){
		return p->x==x&&p->y==y;
	}
};

class Street {
public:

	unsigned int id = 0;
	Point *start = NULL;
	Point *end = NULL;
	double length = -1.0;//Euclid distance of vector start->end
	vector<Street *> connected;

	Street *father_from_origin = NULL;
	double dist_from_origin = 0;

	void print(){
		printf("%ld\t: ",id);
		printf("[[%f,%f],",start->x,start->y);
		printf("[%f,%f]]\t",end->x,end->y);
		printf("\tconnect: ");
		for(Street *s:connected) {
			printf("%ld\t",s->id);
		}
		printf("\n");
	}


	//not distance in real world, but Euclid distance of the vector from start to end
	double getLength() {
		if(length<0) {
			length = start->distance(*end);
		}
		return length;
	}

	Street(unsigned int i, Point *s, Point *e) {
		start = s;
		end = e;
		id = i;
	}
	Street() {

	}

	Point *close(Street *seg) {
		if(seg==NULL) {
			return NULL;
		}
		if(seg->start->equals(start)||seg->start->equals(end)) {
			return seg->start;
		}
		if(seg->end->equals(end)||seg->end->equals(start)) {
			return seg->end;
		}
		return NULL;
	}

	//whether the target segment interact with this one
	//if so, put it in the connected map
	bool touch(Street *seg) {
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
	Street *breadthFirst(long target_id) {

		if(id==target_id) {
			return this;
		}
		queue<Street *> q;
		q.push(this);
		while(!q.empty()) {

			Street *s = q.back();
			q.pop();
			if(s->id == target_id) {//found
				return s;
			}
			for(Street *sc:s->connected) {
				if(sc==this) {//skip current
					continue;
				}
				if(sc->father_from_origin==NULL) {
					sc->father_from_origin = s;
					q.push(sc);
				}
			}
		}

		return NULL;//not found
	}
};



inline double distance_point_to_segment(double x, double y,
										double x1, double y1,
										double x2, double y2) {
	//the segment is vertical
	if(x1==x2) {
		if(y>max(y1, y2)) {
			return sqrt((x-x1)*(x-x1)+(y-max(y1, y2))*(y-max(y1, y2)));
		}else if(y<min(y1, y2)){
			return sqrt((x-x1)*(x-x1)+(min(y1, y2)-y)*(min(y1, y2)-y));
		}else {
			return abs(x-x1);
		}
	}

	//the segment is horizontal
	if(y1==y2) {
		if(x>max(x1, x2)) {
			return sqrt((y-y1)*(y-y1)+(x-max(x1, x2))*(x-max(x1, x2)));
		}else if(x<min(x1, x2)){
			return sqrt((y-y1)*(y-y1)+(min(x1, x2)-x)*(min(x1, x2)-x));
		}else {
			return abs(y-y1);
		}
	}


	double a = (y1-y2)/(x1-x2);
	double b = y1 - a*x1;
	double a1 = -1*(1/a);
	double b1 = y-a1*x;

	double nx = (b1-b)/(a-a1);
	double ny = a1*nx+b1;
	//the cross point is outside the segment
	if(nx>max(x1, x2)||nx<min(x1, x2)) {
		return sqrt(min((x-x1)*(x-x1)+(y-y1)*(y-y1), (x-x2)*(x-x2)+(y-y2)*(y-y2)));
	}else {
		return sqrt((nx-x)*(nx-x)+(ny-y)*(ny-y));
	}

}





inline double distance_point_to_segment(Point *p, Street *s){
	return distance_point_to_segment(p->x,p->y,s->start->x,s->start->y,s->end->x,s->end->y);
}

#endif /* DATAGEN_GEOMETRY_H_ */
