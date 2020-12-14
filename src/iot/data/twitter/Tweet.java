package iot.data.twitter;

import org.json.JSONObject;

public class Tweet extends iot.common.Event{
	public boolean valid = false;
	public Tweet(String s) {
		JSONObject object = new JSONObject(s);
		if(object.has("coordinates")) {
			if(object.get("coordinates")!=null) {
				System.out.println(object.get("coordinates"));
			}
		}
	}
	@Override
	public void print() {
		// TODO Auto-generated method stub
		
	}

}
