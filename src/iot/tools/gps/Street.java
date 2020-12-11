package iot.tools.gps;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Queue;

import iot.common.Point;
import iot.tools.utils.Util;

/*
 * represents a segment with some features
 * 
 * */
public class Street {
	public Point start;
	public Point end;
	public double length = -1.0;//Euclid distance of vector start->end
	public long id;
	public ArrayList<Street> connected = new ArrayList<Street>();
	public ArrayList<Long> connected_id = new ArrayList<Long>();//internal use only, facilitate list for loading connection relations
	
	public Street father_from_origin;
	public double dist_from_origin;
	
	public void print() {
		System.out.print(id+"\t: ");
		System.out.print("[["+start.longitude+","+start.latitude+"],");
		System.out.println("["+end.longitude+","+end.latitude+"]]\t");
		System.out.print("\tconnect: ");
		
		for(Street s:connected) {
	        System.out.print(s.id+"\t");
		}
	    System.out.println();

	}
	
	
	//not distance in real world, but Euclid distance of the vector from start to end 
	public double getLength() {
		if(length<0) {
			this.length = Util.dist_sqr(start.longitude,start.latitude,end.longitude,end.latitude);
		}
		return this.length;
	}
	
	public Street(Point start, Point end, long id) {
		this.start = start;
		this.end = end;
		this.id = id;
		this.length = Util.dist_sqr(start.longitude,start.latitude,end.longitude,end.latitude);
	}
	
	public Street() {
		// TODO Auto-generated constructor stub
	}

	public Point close(Street seg) {
		if(seg==null) {
			return null;
		}
		if(seg.start.equals(start)||seg.start.equals(end)) {
			return seg.start;
		}
		if(seg.end.equals(end)||seg.end.equals(start)) {
			return seg.end;
		}
		return null;
	}
	
	//whether the target segment interact with this one
	//if so, put it in the connected map
	public boolean touch(Street seg) {
		//if those two streets are connected, record the connection relationship
		//since one of the two streets is firstly added, it is for sure it is unique in others list
		if(close(seg)!=null) {
			connected.add(seg);
			seg.connected.add(this);
			return true;
		}
		return false;		
	}
	
	
	
	/*
	 * commit a breadth-first search start from this
	 * 
	 * */
	public Street breadthFirst(Long target_id) {
		
		if(this.id==target_id) {
			return this;
		}
		Queue<Street> queue = new LinkedList<>();
		queue.add(this);
		while(!queue.isEmpty()) {
			
			Street s = queue.poll();
			if(s.id == target_id) {//found
				return s;
			}
			for(Street sc:s.connected) {
				if(sc==this) {//skip current 
					continue;
				}
				if(sc.father_from_origin==null) {
					sc.father_from_origin = s;
					queue.add(sc);
				}
			}			
		}
		
		return null;//not found
	}
}
