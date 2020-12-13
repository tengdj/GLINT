package iot.streamers;

import org.apache.flink.api.common.state.MapState;
import org.apache.flink.api.common.state.MapStateDescriptor;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.functions.KeyedProcessFunction;
import org.apache.flink.util.Collector;
import org.json.JSONObject;

import iot.common.Event;

public class ChicagoTaxiStreamer extends BaseStreamer{
	@Override
    public void run(){
		DataStream<String> text = env.socketTextStream(host, port, "\n");
		System.out.println("accepting data from "+host+":"+port);
		DataStream<Tuple2<String, String>> processed = text
		    .map(new EventMapper())
		    .keyBy(e->e.geohash)
		    .process(new TaxiStreamProcessor());
	
		processed.print();
	
		try {
		    env.execute("Taxi Streamer");
		} catch (Exception e) {
		    e.printStackTrace();
		}
    }
    
    private class TaxiStreamProcessor
		extends KeyedProcessFunction<String, Event, Tuple2<String, String>> {
		private static final long serialVersionUID = 1L;
		MapState<String, Integer> temp;

		
		@Override
	    public void open(Configuration conf) {
	        // register state handle
		    MapStateDescriptor<String, Integer> descr =
		    		new MapStateDescriptor<>("temp", String.class, Integer.class);
	        temp = getRuntimeContext().getMapState(descr);
	    }
	
	    @Override
	    public void processElement(
				   Event event,
				   Context ctx,
				   Collector<Tuple2<String, String>> out) throws Exception {
	    	
	    	JSONObject obj = event.toJson();
		    String geohash = obj.getString("geohash");
		    if(!temp.contains(geohash)) {
		    	temp.put(geohash, 1);
		    }else {
		    	temp.put(geohash, temp.get(geohash)+1);
		    }
		    out.collect(Tuple2.of(ctx.getCurrentKey(), temp.get(geohash).toString()));
	    	return;
	    }
    }
	
}
