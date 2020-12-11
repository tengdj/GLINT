package iot.common;

import org.json.JSONObject;
import iot.tools.geohash.GeoHash;

public class Event {

	public long timestamp;
	public Point coordinate;
	public JSONObject features = null;
	public String geohash = null;
	public String id = "";
	
	
	public Event() {
		
	}
	public Event(String str) {
		JSONObject obj = new JSONObject(str);
		id = obj.getString("id");
		timestamp = obj.getLong("timestamp");
		coordinate = new Point(obj.getDouble("longitude"),obj.getDouble("latitude"));
		features = obj.getJSONObject("features");
		getGeoHash();
	}
	public JSONObject getFeatures() {
		if(features!=null) {
			return features;
		}
		features = new JSONObject();
		return features;
	}
	
	public long getTime() {
		return timestamp;
	}
	
	public Point getLocation() {
		return coordinate;
	}
	
	public String getGeoHash() {
		if(geohash==null) {
			geohash = GeoHash.withCharacterPrecision(coordinate.latitude, coordinate.longitude, 12).toBase32();
		}
		return geohash;
	}
	
	public JSONObject toJson() {
		JSONObject jsonobj = new JSONObject();
		jsonobj.put("id", id);
		jsonobj.put("timestamp", timestamp);
		jsonobj.put("longitude", coordinate.longitude);
		jsonobj.put("latitude", coordinate.latitude);
		jsonobj.put("geohash", getGeoHash());
		jsonobj.put("features", getFeatures());
		return jsonobj;
	}
	
	public String toString() {
		return toJson().toString();
	}
	public String toString(int dent) {
		return toJson().toString(dent);
	}
	
	public void print() {
		System.out.println(toString());
	};
	
	
}
