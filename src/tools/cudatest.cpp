/*
 * cudatest.cpp
 *
 *  Created on: Jan 25, 2021
 *      Author: teng
 */


#include "../cuda/mygpu.h"
#include "../util/util.h"
#include "../util/query_context.h"
#include "../geometry/geometry.h"

int foo(Point *p1, Point *p2);

int main(int argc, char **argv){
	Point p1(-87.607680,41.892117);
	Point p2(-87.607769,41.892187);
	cout<<"cpu "<<p1.distance(p2,true)<<endl;
	foo(&p1,&p2);


}

