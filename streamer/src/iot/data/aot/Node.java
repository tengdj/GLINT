package iot.data.aot;

import java.text.ParseException;

import iot.tools.geohash.GeoHash;
import iot.tools.utils.Util;

public class Node {
	
	public long node_id;
	public String project_id;
	public String vsn;
	public String address;
	public double latitude;
	public double longitude;
	public String description;
	public Long start_timestamp = null;
	public Long end_timestamp = null;
	public String geohash;
	
	public Node(String data[]) throws ParseException {
		
//		for(int i=0;i<data.length;i++) {
//			System.out.println(i+"--"+data[i]);
//		}
		node_id = Long.parseLong(data[0],16);
		project_id = data[1];
		vsn = data[2];
		address = data[3];
		latitude = Double.parseDouble(data[4]);
		longitude = Double.parseDouble(data[5]);
		geohash = GeoHash.withCharacterPrecision(latitude, longitude, 12).toBase32();
		description = data[6];
		if(!data[7].isEmpty()) {
			start_timestamp = Util.getTimestamp(data[7]);
		}
		if(data.length==9&&!data[8].isEmpty()) {
			end_timestamp = Util.getTimestamp(data[8]);
		}
	}

}
