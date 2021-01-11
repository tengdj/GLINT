package iot.tools.gps;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

import org.json.JSONArray;
import org.json.JSONObject;

import iot.common.Point;
import iot.tools.utils.Util;

/*
 * class for the streets data 
 * 
 * */
public class Map {
	
	protected ArrayList<Street> streets = new ArrayList<Street>();
	
	
	public Map(){
		
	}
	
	public void clear() {
		for(Street s:streets) {
			s.connected.clear();
		}
		streets.clear();
	}
	
	public ArrayList<Street> getStreets(){
		return streets;
	}
	
	/*
	 * compare each street pair to see if they connect with each other
	 * */
	protected void connect_segments() {
		System.out.println("connecting streets");
		for(int i=0;i<streets.size()-1;i++) {
			for(int j=i+1;j<streets.size();j++) {
				streets.get(i).touch(streets.get(j));
			}
		}		
	}
	
	private static JSONObject wrapGeoJSON(ArrayList<JSONObject> origins) {
		JSONObject col = new JSONObject();
		col.put("type","FeatureCollection");
		JSONArray feas = new JSONArray();
		int i = 0;
		for(JSONObject o:origins) {
			JSONObject obj = new JSONObject();
			obj.put("type", "Feature");
			obj.put("geometry", o);
			JSONObject ps = new JSONObject();
			ps.put("id", ++i);
			obj.put("properties",ps);
			feas.put(obj);
		}
		col.put("features", feas);
		return col;
	}
	
	public static JSONObject genGeoJson(ArrayList<Street> streets) {
		
		
		JSONObject obj = new JSONObject();
		
		obj.put("type", "MultiLineString");
		
		JSONArray arr = new JSONArray();
		
		StringBuilder sb = new StringBuilder();
		sb.append("{\"type\":\"MultiLineString\",\"coordinates\":\n\t[\n");
		for(Street s:streets) {
			JSONArray sta = new JSONArray();
			JSONArray jas = new JSONArray();
			JSONArray jae = new JSONArray();
			jas.put(s.start.longitude);
			jas.put(s.start.latitude);
			jae.put(s.end.longitude);
			jae.put(s.end.latitude);
			sta.put(jas);
			sta.put(jae);
			arr.put(sta);
		}
		obj.put("coordinates", arr);
		
		ArrayList<JSONObject> ret = new ArrayList<JSONObject>();
		ret.add(obj);
		return wrapGeoJSON(ret);
	}
	
	public static JSONObject genGeoJsonPoints(ArrayList<Point> points) {
		
		ArrayList<JSONObject> ret = new ArrayList<JSONObject>();
		for(Point p:points) {
			JSONObject o = new JSONObject();
			o.put("type", "Point");
			JSONArray co = new JSONArray();
			co.put(p.longitude);
			co.put(p.latitude);
			o.put("coordinates", co);
			ret.add(o);
		}
		return wrapGeoJSON(ret);
	}
	
	public String toString() {
		return Map.genGeoJson(streets).toString();
	}
	public void print() {
		
		System.out.println(toString());

	}
	
	public void dumpTo(String path) {
		try {
			
			DataOutputStream stream = new DataOutputStream(new FileOutputStream(path));
			stream.writeInt(streets.size());
			for(Street street:streets) {
				stream.writeLong(street.id);
				stream.writeDouble(street.start.longitude);
				stream.writeDouble(street.start.latitude);
				stream.writeDouble(street.end.longitude);
				stream.writeDouble(street.end.latitude);
				stream.writeInt(street.connected.size());
				for(Street s:street.connected) {
			        stream.writeLong(s.id);
				}

			}
			stream.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public void toGMSF(String path) {
		try {
		  FileWriter myWriter = new FileWriter(path);
		  myWriter.write("Files in Java might be tricky, but it is fun enough!");
		  
		  for(Street street:streets) {
			  myWriter.write("<Road>\n");
			  myWriter.write(street.id+" 0 "+street.start.longitude+" "+street.start.latitude+" "+street.end.longitude+" "+street.end.latitude+"\n");
			  myWriter.write("</Road>\n");
		  }

		  myWriter.close();
		} catch (IOException e) {
		  System.out.println("An error occurred.");
		  e.printStackTrace();
		}
	}
	
	
	
	public void loadFromFormatedData(String path) {
		
		System.out.println("loading from formated file "+path);
		HashMap<Long, Street> stmap = new HashMap<Long, Street>();
		try {
			DataInputStream stream = new DataInputStream(new FileInputStream(path));
			int size = stream.readInt();
			for(int i=0;i<size;i++) {
				Street s = new Street();
				s.id = stream.readLong();
				s.start = new Point(stream.readDouble(),stream.readDouble());
				s.end = new Point(stream.readDouble(),stream.readDouble());
				int connected_size = stream.readInt();
				for(int j=0;j<connected_size;j++) {
					s.connected_id.add(stream.readLong());
				}
				streets.add(s);
				stmap.put(s.id, s);
			}
			stream.close();
			
			//now map the id to object
			for(Street s:streets) {
				for(Long sid:s.connected_id) {
					s.connected.add(stmap.get(sid));
				}
			}
			System.out.println(size+" streets are loaded");
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
	
	
	
	double distPointToSegment(Point p, Street s){
		return Util.min_distance_point_to_segment(p.longitude, p.latitude, s.start.longitude, s.start.latitude, s.end.longitude, s.end.latitude);
	}
	
	
	public ArrayList<Street> nearest(Point target, int limit){
		ArrayList<Street> ret = new ArrayList<Street>();
		ArrayList<Double> dist = new ArrayList<Double>();
		double min = Double.MAX_VALUE;
		for(Street st:streets) {
			double d = this.distPointToSegment(target, st);
			if(dist.size()==0) {
				min = d;
				dist.add(d);
				ret.add(st);
				continue;
			}
			//the queue is full and the distance is bigger than or equal to the current minimum
			if(dist.size()>=limit&&d>=min) {
				continue;
			}
			//otherwise, insert current street into the return list, evict the tail
			int insert_into = 0;
			for(;insert_into<dist.size();insert_into++) {
				if(dist.get(insert_into)>=d) {
					ret.add(insert_into, st);
					dist.add(insert_into, d);
					break;
				}
			}
			
			if(ret.size()>limit) {
				ret.remove(limit);
				dist.remove(limit);
			}
			min = dist.get(dist.size()-1);
			
		}
		dist.clear();
		return ret;
	}
	
	public ArrayList<Street> navigate(Point origin, Point dest){
		
		ArrayList<Street> ret = new ArrayList<Street>();
		ArrayList<Street> originset = this.nearest(origin, 5);
		ArrayList<Street> destset = this.nearest(dest, 5);
		
		for(Street o:originset) {
			for(Street d:destset) {
				//initialize 
				for(Street s:streets) {
					s.father_from_origin = null;
				}
				Street s = o.breadthFirst(d.id);
				if(s!=null) {
					while(s.father_from_origin!=null) {
						ret.add(s);
						s = s.father_from_origin;
					}
					break;
				}
			}
			
			if(ret.size()>0) {
				break;
			}
		}
		
		ArrayList<Street> reversed = new ArrayList<Street>();
		reversed.ensureCapacity(ret.size());
		for(int i=ret.size()-1;i>=0;i--) {
			reversed.add(ret.get(i));
		}
		ret.clear();
		return reversed;
		
	}

}
