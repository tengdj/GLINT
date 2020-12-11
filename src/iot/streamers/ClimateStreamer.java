package iot.streamers;
/* Flink imports */
import org.apache.flink.util.Collector;
import org.json.JSONObject;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.api.common.state.MapState;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.api.common.state.MapStateDescriptor;
import org.apache.flink.streaming.api.functions.KeyedProcessFunction;
/* Local imports */
import iot.common.Event;
import iot.tools.utils.Util;

public class ClimateStreamer extends BaseStreamer {

    @Override
    public void run(){
		DataStream<String> text = env.socketTextStream(host, port, "\n");
		DataStream<Tuple2<String, String>> processed = text
		    .map(new EventMapper())
		    .keyBy(e->e.geohash)
		    .process(new ClimateStreamProcessor());
	
		processed.print();
	
		try {
		    env.execute("Climate Streamer");
		} catch (Exception e) {
		    e.printStackTrace();
		}
    }
    
    private class ClimateStreamProcessor
		extends KeyedProcessFunction<String, Event, Tuple2<String, String>> {
    	
    	private class Temperature{
        	public double min;
        	public double max;
        	public long min_time;
        	public long max_time;
        	public Temperature(double min, long min_time, double max, long max_time) {
        		this.min = min;
        		this.max = max;
        		this.min_time = min_time;
        		this.max_time = max_time;
        	}
        }
    	
        private final Double temperature_threshold = 20.0;
        //tolerate a gap as long as 5 days
        private final long time_gap = 5*24*60*60*1000;
		private static final long serialVersionUID = 1L;
		MapState<String, Temperature> temp;

		
		@Override
	    public void open(Configuration conf) {
	        // register state handle
		    MapStateDescriptor<String, Temperature> descr =
		    		new MapStateDescriptor<>("temp", String.class, Temperature.class);
	        temp = getRuntimeContext().getMapState(descr);
	    }
	
	    @Override
	    public void processElement(
				   Event event,
				   Context ctx,
				   Collector<Tuple2<String, String>> out) throws Exception {
	    	
	    	JSONObject obj = event.toJson();
	    	
		    String element = event.getFeatures().getString("element");
		    //focus on minimum and maximum temperature
		    JSONObject feature = event.getFeatures();
		    double value = feature.getDouble("value");
		    long timestamp = obj.getLong("timestamp");
		    String geohash = obj.getString("geohash");
		    if(!temp.contains(geohash)) {
		    	temp.put(geohash, new Temperature(value, timestamp, value, timestamp));
		    	return;
		    }
		    
		    Temperature cur_temp = temp.get(geohash);
		    // alert if TMIN drop too much
		    if(element.contentEquals("TMIN")) {
		    	// drop in one day
		    	if((timestamp-cur_temp.min_time<=time_gap)
		    			&&(cur_temp.min-value>=temperature_threshold)) {
		    		String alert = "station "+feature.getString("stationid")+" drop from "+cur_temp.min+" to "
		    				+value+" at "+Util.formatTimestamp(timestamp);
		    		out.collect(Tuple2.of(ctx.getCurrentKey(), alert));
		    	}
		    	cur_temp.min = value;
		    	cur_temp.min_time = timestamp;
		    }else if(element.contentEquals("TMAX")) {
		    	// drop in one day
		    	if((timestamp-cur_temp.max_time<=this.time_gap)
		    			&&(value-cur_temp.max>=temperature_threshold)) {
		    		String alert = "station "+feature.getString("stationid")+" increase from "+cur_temp.max+" to "
		    				+value+" at "+Util.formatTimestamp(timestamp);
		    		out.collect(Tuple2.of(ctx.getCurrentKey(), alert));
		    	}
		    	cur_temp.max = value;
		    	cur_temp.max_time = timestamp;
		    }
		    
		    //update the state container
		    temp.put(geohash, cur_temp);
	    }
    }

}
