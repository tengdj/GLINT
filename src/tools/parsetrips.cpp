///*
// * parsetrips.cpp
// *
// *  Created on: Mar 2, 2021
// *      Author: teng
// */
//
//
//#include "../tracing/trace.h"
//
//#include "../geometry/Map.h"
//#include "../util/config.h"
//#include "../tracing/generator.h"
//#include <vector>
//#include <stdlib.h>
//
//using namespace std;
//int main(int argc, char **argv){
//
//	struct timeval start = get_cur_time();
//
//	generator_configuration config = get_generator_parameters(argc, argv);
//	box mbr(-87.985453, 41.644584, -87.479161, 42.022927);
//	mbr.print();
//
//	cout<<194695786*sizeof(Trip)/1024/1024<<endl;
//
//	std::ifstream file(config.taxi_path);
//	if(!file.is_open()){
//		log("%s cannot be opened",config.taxi_path.c_str());
//		exit(0);
//	}
//
//	map<pair<double, double>, uint> coordid;
//	map<pair<uint, uint>, uint> trip_counter;
//
//	uint counter = 0;
//	std::string str;
//	//skip the head
//	std::getline(file, str);
//	while (std::getline(file, str)){
//		Trip *t = new Trip(str);
//		if(mbr.contain(t->start.coordinate)&&mbr.contain(t->end.coordinate)){
//			pair<double,double> p1(t->start.coordinate.x,t->start.coordinate.y);
//			pair<double,double> p2(t->end.coordinate.x,t->end.coordinate.y);
//			uint id1 = 0;
//			uint id2 = 0;
//			if(coordid.find(p1)!=coordid.end()){
//				id1 = coordid[p1];
//			}else{
//				id1 = coordid.size();
//				coordid[p1] = id1;
//			}
//			if(coordid.find(p2)!=coordid.end()){
//				id2 = coordid[p2];
//			}else{
//				id2 = coordid.size();
//				coordid[p2] = id2;
//			}
//			trip_counter[pair<uint,uint>(id1,id2)]++;
//		}
//		if(++counter%1000000==0){
//			cerr<<counter<<" "<<coordid.size()<<" "<<trip_counter.size()<<endl;
//			//break;
//		}
////		t->start.coordinate.x += 0.02*(get_rand_double()-0.5);
////		t->start.coordinate.y += 0.012*(get_rand_double()-0.5);
////		t->end.coordinate.x += 0.02*(get_rand_double()-0.5);
////		t->end.coordinate.y += 0.012*(get_rand_double()-0.5);
//		// a valid trip should be covered by the map,
//		// last for a while and the distance larger than 0
//		delete t;
//	}
//	file.close();
//
//	cout<<coordid.size()<<endl;
//	for(auto a:coordid){
//		printf("%f %f %d\n",a.first.first, a.first.second, a.second);
//	}
//	cout<<trip_counter.size()<<endl;
//	for(auto a:trip_counter){
//		printf("%d %d %d\n",a.first.first, a.first.second, a.second);
//	}
//
//	coordid.clear();
//	trip_counter.clear();
//	logt("parse",start);
//	return 0;
//}
//
//
//
