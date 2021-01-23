/*
 * geometry.h
 * included the most commonly accessed data structures
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
#include <queue>
#include <iostream>
#include "util/util.h"

using namespace std;
class Point{
public:
	double x = 0;
	double y = 0;
	Point(double xx, double yy){
		x = xx;
		y = yy;
	}
	Point(){}
	Point(Point *p){
		x = p->x;
		y = p->y;
	}
	~Point(){}
	double distance(Point &p, bool geography = false){
		double dx = x-p.x;
		double dy = y-p.y;
		if(geography){
			dy = dy/degree_per_kilometer_latitude;
			dx = dx/degree_per_kilometer_longitude(y);
		}
		return sqrt(dx*dx+dy*dy);
	}
	bool equals(Point &p){
		return p.x==x&&p.y==y;
	}
	void print(){
		printf("POINT (%f %f)\n",x,y);
	}

};

class box{
public:
	double low[2] = {100000.0,100000.0};
	double high[2] = {-100000.0,-100000.0};

	box(){}

	box(box *b){
		low[0] = b->low[0];
		high[0] = b->high[0];
		low[1] = b->low[1];
		high[1] = b->high[1];
	}

	void update(Point &p){
		if(low[0]>p.x){
			low[0] = p.x;
		}
		if(high[0]<p.x){
			high[0] = p.x;
		}

		if(low[1]>p.y){
			low[1] = p.y;
		}
		if(high[1]<p.y){
			high[1] = p.y;
		}
	}

	bool intersect(box &target){
		return !(target.low[0]>high[0]||
				 target.high[0]<low[0]||
				 target.low[1]>high[1]||
				 target.high[1]<low[1]);
	}
	bool contain(box &target){
		return target.low[0]>=low[0]&&
			   target.high[0]<=high[0]&&
			   target.low[1]>=low[1]&&
			   target.high[1]<=high[1];
	}
	bool contain(Point &p){
		return p.x>=low[0]&&
			   p.x<=high[0]&&
			   p.y>=low[1]&&
			   p.y<=high[1];
	}



	double area(bool geography = false){
		if(!geography){
			return (high[0]-low[0])*(high[1]-low[1]);
		}else{
			double h = (high[1]-low[1])/degree_per_kilometer_latitude;
			double top = (high[0]-low[0])/degree_per_kilometer_longitude(high[1]);
			double bot = (high[0]-low[0])/degree_per_kilometer_longitude(low[1]);
			printf("%f %f %f\n",h,top,bot);
			return (top+bot)*h/2;
		}
	}



	double distance(Point &p, bool geography = false){
		if(this->contain(p)){
			return 0;
		}
		double dx = max(abs(p.x-(low[0]+high[0])/2) - (high[0]-low[0])/2, 0.0);
		double dy = max(abs(p.y-(low[1]+high[1])/2) - (high[1]-low[1])/2, 0.0);
		if(geography){
			dy = dy/degree_per_kilometer_latitude;
			dx = dx/degree_per_kilometer_longitude(p.y);
		}
		return sqrt(dx * dx + dy * dy);
	}


	double max_distance(Point &p){
		double md = 0;
		double dist = (p.x-low[0])*(p.x-low[0])+(p.y-low[1])*(p.y-low[1]);
		if(dist>md){
			md = dist;
		}
		dist = (p.x-low[0])*(p.x-low[0])+(p.y-high[1])*(p.y-high[1]);
		if(dist>md){
			md = dist;
		}
		dist = (p.x-high[0])*(p.x-high[0])+(p.y-low[1])*(p.y-low[1]);
		if(dist>md){
			md = dist;
		}
		dist = (p.x-high[0])*(p.x-high[0])+(p.y-high[1])*(p.y-high[1]);
		if(dist>md){
			md = dist;
		}
		return sqrt(md);
	}

	void print_vertices(){
		printf("%f %f, %f %f, %f %f, %f %f, %f %f",
					low[0],low[1],
					high[0],low[1],
					high[0],high[1],
					low[0],high[1],
					low[0],low[1]);
	}

	void print(){
		printf("POLYGON((");
		print_vertices();
		printf("))\n");

	}

	void to_squre(){
		int bigger_one = (high[1]-low[1]>high[0]-low[0]);
		double difference = (high[bigger_one]-low[bigger_one])-(high[!bigger_one]-low[!bigger_one]);
		low[!bigger_one] = low[!bigger_one]-difference/2;
		high[!bigger_one] = high[!bigger_one]+difference/2;
	}
};

/*
 * some utility functions shared by other classes
 *
 * */

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


inline void print_linestring(vector<Point *> trajectory, double sample_rate=1.0){
	assert(sample_rate<=1&&sample_rate>0);
	printf("LINESTRING (");
	bool first = true;
	for(int i=0;i<trajectory.size();i++){

		if(tryluck(sample_rate)){
			if(!first){
				printf(",");
			}else{
				first = false;
			}
			printf("%f %f",trajectory[i]->x,trajectory[i]->y);
		}
	}

	printf(")\n");
}

inline void print_points(vector<Point *> trajectory, double sample_rate=1.0){
	assert(sample_rate<=1&&sample_rate>0);
	printf("MULTIPOINT (");
	bool first = true;
	for(int i=0;i<trajectory.size();i++){

		if(tryluck(sample_rate)){
			if(!first){
				printf(",");
			}else{
				first = false;
			}
			printf("%f %f",trajectory[i]->x,trajectory[i]->y);
		}
	}

	printf(")\n");
}


#endif /* DATAGEN_GEOMETRY_H_ */
