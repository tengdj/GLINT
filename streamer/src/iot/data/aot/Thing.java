package iot.data.aot;

import java.text.ParseException;

import org.json.JSONObject;

import iot.common.Event;
import iot.tools.utils.Util;

public class Thing extends Event{

	public long node_id;
	public String subsystem;
	public String sensor;
	public String parameter;
	public Double value_raw = null;
	public Double value_hrf = null;
	
	//parse data from a line of data, split with comma
	public Thing(String data[]) throws ParseException {
		
		timestamp = Util.getTimestamp(data[0]);

		node_id = Long.parseLong(data[1],16);
		subsystem = data[2];
		sensor = data[3];
		parameter = data[4];
		if(!data[5].isEmpty()&&!data[5].contentEquals("NA")) {
			if(data[5].contains(".")) {//from float number
				value_raw = Double.parseDouble(data[5]);
			}else if(data[5].charAt(0)=='0'){//hex
				value_raw = (double)(long)Long.parseLong(data[5], 16);
			}else {//integer
				try {
					value_raw = (double)(long)Long.parseLong(data[5]);
				}catch(Exception e) {//
					
				}
			}
		}
		
		if(!data[6].isEmpty()&&!data[6].contentEquals("NA")) {
			if(data[6].contains(".")) {//from float number
				value_hrf = Double.parseDouble(data[6]);
			}else if(data[6].charAt(0)=='0'){//hex
				value_hrf = (double)(long)Long.parseLong(data[6], 16);
			}else {//integer
				try {
					value_hrf = (double)(long)Long.parseLong(data[6]);
				}catch(Exception e) {//
					
				}
			}
		}
		
	}
	

	@Override
	public JSONObject getFeatures() {
		JSONObject obj = new JSONObject();
		obj.put("node_id", subsystem);
		obj.put("sensor", sensor);
		obj.put("parameter", parameter);
		obj.put("value_raw", value_raw);
		obj.put("value_hrf", value_hrf);

		return obj;
	}

	@Override
	public void print() {
		// TODO Auto-generated method stub
		
	}

}
