package iot.streamers;
/* Flink imports */
import org.apache.flink.util.Collector;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.api.common.state.MapState;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.api.common.state.MapStateDescriptor;
import org.apache.flink.streaming.api.functions.KeyedProcessFunction;
/* Local imports */
import iot.common.Event;

import org.apache.flink.streaming.api.windowing.time.Time;

public class AOTStreamer extends BaseStreamer {
    
    private final String outPath = "/home/cloudera/Downloads/IoTDB/streamer/processed_out.txt";
    
    @Override
	public void run(){
		DataStream<String> text = env.socketTextStream(host, port, "\n");
//		DataStream<Tuple2<String, String>> processed = text
//		    .map(new EventMapper())
//		    .keyBy(e->e.geohash)
//		    .process(new AOTStreamProcessor());
//		
//		processed.print();
//		processed.writeAsText("outPath");
//		text.map(new ThingMapper()).writeAsText(outPath);
	
		DataStream<Tuple2<String, Long>> count = text                                                            
		    .flatMap(new SimpleMapper())                                                                         
		    //.keyBy(0)                                                                                          
		    .timeWindowAll(Time.seconds(1))                                                                      
		    .reduce(new SimpleReducer());                                                                        
		
		count.writeAsText(outPath);
		
		try {
		    env.execute("AOT Streamer");
		} catch (Exception e) {
		    e.printStackTrace();
		}
    }
    
    private class AOTStreamProcessor
	extends KeyedProcessFunction<String, Event, Tuple2<String, String>> {
    	
        private final Double threshold = 2.0;
		private static final long serialVersionUID = 1L;
		MapState<String, Double> locationParams;
			
		@Override
	    public void open(Configuration conf) {
		    // register state handle
		    MapStateDescriptor<String, Double> descr =
			new MapStateDescriptor<>("parameters", String.class, Double.class);
		    locationParams = getRuntimeContext().getMapState(descr);
		}
		
		@Override
	    public void processElement(
				       Event event,
				       Context ctx,
				       Collector<Tuple2<String, String>> out) throws Exception {
		
		    if (event.getFeatures().isNull("value_raw")) {
		    	return;
		    }
			    
		    String eventParam = event.getFeatures().getString("parameter");
		    Double eventParamVal = event.getFeatures().getDouble("value_raw");
		
		    // check if event value is greater than specified threshold
		    if (locationParams.contains(eventParam)){
				Double currParamVal = locationParams.get(eventParam);
				Double delta = Math.abs(currParamVal - eventParamVal);
				if (delta > threshold){
				    String alert = "Parameter: " + eventParam + ", " +
					"last value: " + currParamVal + ", " +
					"new value: " + eventParamVal + " " +
					"(delta: " + delta + ")";
						    
				    out.collect(Tuple2.of(ctx.getCurrentKey(), alert));
				}
		    }
			    
		    // insert/update parameter value in state
		    locationParams.put(eventParam, eventParamVal);
		}
    }

}
