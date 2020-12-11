package iot.data.taxi;

import java.util.ArrayList;

import iot.common.Point;
import iot.tools.gps.Street;
import iot.tools.utils.Util;

public class Trip {

	//note that the time in the Chicago taxi dataset is messed up
	//the end_time and start_time is rounded to the nearest 15 minute like 0, 15, 45.
	//the duration_time is rounded to the nearest 10 seconds
	//thus for most cases end_time-start_time != duration_time
	public long start_time;
	public long end_time;
	public double duration_time;
	Point start_location;
	Point end_location;
	public double trip_length;
	public Trip(String cols[]) throws Exception {
		
		
		start_time = Util.getTimestamp("MM/dd/yyyy hh:mm:ss a", cols[2]);
		end_time = Util.getTimestamp("MM/dd/yyyy hh:mm:ss aa",cols[3]);
		duration_time = (double) Integer.parseInt(cols[4]);
		if(duration_time<=0) {
			throw new Exception();
		}
		trip_length = Double.parseDouble(cols[14]);
		start_location = new Point(Double.parseDouble(cols[18]),Double.parseDouble(cols[17]));
		end_location = new Point(Double.parseDouble(cols[21]),Double.parseDouble(cols[20]));

	}
	
	public void print() {
		System.out.println((end_time-start_time)/1000+"\t"+trip_length+"\t"+start_location.longitude+","+start_location.latitude+"\t"+ end_location.longitude+","+end_location.latitude);
	}
	
	/*
	 * simulate the trajectory of the trip. 
	 * with the given streets the trip has covered, generate a list of points 
	 * that the taxi may appear at a given time
	 * 
	 * */
	public ArrayList<CurrentPosition> getCurLocations(ArrayList<Street> st) {
		
		ArrayList<CurrentPosition> positions = new ArrayList<CurrentPosition>();
		
		double total_length = 0;
		for(Street s:st) {
			total_length += s.getLength();
		}
		double step = total_length/duration_time;		
		Point origin = null;
		double dist_from_origin = 0;
		if(st.size()==1) {
			origin = st.get(0).start;
		}else {
			if(st.get(0).start.equals(st.get(1).start)||st.get(0).start.equals(st.get(1).end)) {
				origin = st.get(0).end;
			}else {
				origin = st.get(0).start;
			}
		}
		
		for(Street s:st) {
			boolean from_start = origin.equals(s.start);
			double next_dist_from_origin = dist_from_origin+=s.length;

			Point cur_start = from_start?s.start:s.end;
			Point cur_end = from_start?s.end:s.start;
			
			double cur_dis = ((int)(next_dist_from_origin/step)+1)*step-dist_from_origin;
			while(cur_dis<s.length) {//have other position can be reported in this street
				double cur_portion = cur_dis/s.length;
				//now get the longitude and latitude and timestamp for current event and add to return list
				CurrentPosition cp = new CurrentPosition();
				cp.timestamp = (long)(((cur_dis+dist_from_origin)*1000/total_length)*this.duration_time+this.start_time);
				cp.coordinate = new Point(cur_start.longitude+(cur_end.longitude-cur_start.longitude)*cur_portion,
						cur_start.latitude+(cur_end.latitude-cur_start.latitude)*cur_portion);
				positions.add(cp);
				cur_dis += step;
			}
			
			
			//now cut cur_start->cur_end according to the start and step
			
			//move to next street
			dist_from_origin = next_dist_from_origin;
			if(from_start) {
				origin = s.end;
			}else {
				origin = s.start;
			}
		}	
		
		return positions;
		
	}
	
}
