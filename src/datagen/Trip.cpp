/*
 * Trip.cpp
 *
 *  Created on: Jan 12, 2021
 *      Author: teng
 */

#include "Map.h"



Trip::Trip(string str){

	vector<string> cols;
	tokenize(str,cols,",");

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
	if(cols[2][20]=='P'){
		start_time += 12*3600;
	}
	end_time = start_time + atoi(cols[4].c_str());

	start_location = Point(atof(cols[18].c_str()),atof(cols[17].c_str()));
	end_location = Point(atof(cols[21].c_str()),atof(cols[20].c_str()));
}

void Trip::navigate(Map *m){
	trajectory = m->navigate(&start_location, &end_location);
}

void Trip::print_trip(){
	printf("time: %d to %d\n",start_time,end_time);
	printf("position: (%f %f) to (%f %f)\n",start_location.x,start_location.y,end_location.x,end_location.y);
}

/*
 * simulate the trajectory of the trip.
 * with the given streets the trip has covered, generate a list of points
 * that the taxi may appear at a given time
 *
 * */
vector<Point *> Trip::getTraces(){

	int duration_time = end_time-start_time;

	vector<Point *> positions;

	double total_length = 0;
	for(int i=0;i<trajectory.size()-1;i++){
		total_length += trajectory[i]->distance(*trajectory[i+1]);
	}
	//ever second
	double step = total_length/duration_time;
	double dist_from_origin = 0;
	for(int i=0;i<trajectory.size()-1;i++) {
		Point *cur_start = trajectory[i];
		Point *cur_end = trajectory[i+1];
		double length = cur_start->distance(*cur_end);
		double next_dist_from_origin = dist_from_origin+=length;
		double cur_dis = ((int)(next_dist_from_origin/step)+1)*step-dist_from_origin;
		printf("%f %f\n",cur_dis,length);
		while(cur_dis<length) {//have other position can be reported in this street
			double cur_portion = cur_dis/length;
			//now get the longitude and latitude and timestamp for current event and add to return list
			Point *p = new Point(cur_start->x+(cur_end->x-cur_start->x)*cur_portion,
								 cur_start->y+(cur_end->y-cur_start->y)*cur_portion);
			positions.push_back(p);
			cur_dis += step;
		}

		//move to next street
		dist_from_origin = next_dist_from_origin;
	}

	return positions;

}

vector<Trip *> load_trips(const char *path, int limit){
	std::ifstream file(path);
	std::string str;
	vector<Trip *> trips;
	//skip the head
	std::getline(file, str);
	while (std::getline(file, str)){
		Trip *t = new Trip(str);
		trips.push_back(t);
		if(trips.size()>limit){
			break;
		}
	}
	return trips;

}
