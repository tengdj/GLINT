///*
// * parsetweets.cpp
// *
// *  Created on: Mar 2, 2021
// *      Author: teng
// */
//
//
//#include "../tracing/trace.h"
//#include "../tracing/generator.h"
//
//#include "../geometry/Map.h"
//#include "../util/config.h"
//#include <vector>
//#include <stdlib.h>
//
//bool orderdest(pair<int, double> i, pair<int, double> j) { return (i.second>j.second); }
//
//
//
//using namespace std;
//int main(int argc, char **argv){
//
//	generator_configuration config = get_generator_parameters(argc, argv);
//	box mbr(-87.985453, 41.644584, -87.479161, 42.022927);
//	struct timeval start = get_cur_time();
//	long fsize = file_size(config.tweet_path.c_str());
//	if(fsize<=0){
//		log("%s is empty",config.tweet_path.c_str());
//		exit(0);
//	}else{
//		log("size of %s is %ld",config.tweet_path.c_str(),fsize);
//	}
//	uint target_num = fsize/(2*sizeof(double));
//
//	Point *tweets = new Point[target_num];
//	ifstream infile(config.tweet_path.c_str(), ios::in | ios::binary);
//	infile.read((char *)tweets, fsize);
//	infile.close();
//
//
//	ifstream trip_file(config.taxi_path.c_str(), ios::in);
//	int num_nodes = 0;
//	trip_file>>num_nodes;
//	gen_core *cores = new gen_core[num_nodes];
//	double x,y;
//	int node = 0;
//	for(int i=0;i<num_nodes;i++){
//		trip_file>>x;
//		trip_file>>y;
//		trip_file>>node;
//		cores[node].id = i;
//		cores[node].core.x = x;
//		cores[node].core.y = y;
//	}
//
//	int con_num = 0;
//	trip_file>>con_num;
//	int s = 0;
//	int d = 0;
//	int n = 0;
//	for(int i=0;i<con_num;i++){
//		trip_file>>s;
//		trip_file>>d;
//		trip_file>>n;
//		cores[s].destination.push_back(pair<int,double>(d,(double)n));
//	}
//	for(int i=0;i<num_nodes;i++){
//		sort(cores[i].destination.begin(),cores[i].destination.end(),orderdest);
//		double total = 0;
//		for(pair<int,double> &p:cores[i].destination){
//			total += p.second;
//		}
//		for(pair<int,double> &p:cores[i].destination){
//			p.second = p.second/total;
//		}
//	}
//	ofstream out("chicago.mt");
//	out.write((char *)&target_num, sizeof(target_num));
//	for(int i=0;i<target_num;i++){
//		out.write((char *)&tweets[i], sizeof(Point));
//		double min_dist = DBL_MAX;
//		int min_core = 0;
//		// get the closest
//		for(int j=0;j<num_nodes;j++){
//			double dist = cores[j].core.distance(tweets[i]);
//			if(dist<min_dist&&cores[j].destination.size()>0){
//				min_dist = dist;
//				min_core = j;
//			}
//		}
//		out.write((char *)&min_core, sizeof(int));
//		cores[min_core].assigned_tweets.push_back(i);
//	}
//
//	out.write((char *)&num_nodes, sizeof(num_nodes));
//	for(int i=0;i<num_nodes;i++){
//		out.write((char *)&cores[i].core, sizeof(Point));
//		int num_dest = cores[i].destination.size();
//		out.write((char *)&num_dest, sizeof(num_dest));
//		for(int j=0;j<num_dest;j++){
//			out.write((char *)&cores[i].destination[j].first, sizeof(int));
//			out.write((char *)&cores[i].destination[j].second, sizeof(double));
//		}
//	}
//	out.close();
//	//print_points(points,target_num);
//	logt("parsed", start);
//	return 0;
//}
//
//
//
