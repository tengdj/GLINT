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

