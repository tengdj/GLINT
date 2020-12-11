package iot.streamers;

import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.common.state.ValueState;
import org.apache.flink.api.common.state.ValueStateDescriptor;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.KeyedProcessFunction;
import org.apache.flink.streaming.api.windowing.time.Time;
import org.apache.flink.util.Collector;
import org.json.JSONObject;

import iot.common.Event;
import iot.tools.utils.StreamerConfig;


/*
 * 
 * the basic Streamer class which implements two basic functions
 * globalCount to count the events for each geohash tag
 * windowCount to count the events for each geohash tag in the past time window
 * 
 * */
public class BaseStreamer extends Thread{
	
    int port = StreamerConfig.getInt("stream-port");
    String host = StreamerConfig.get("stream-host");
	
	public void setPort(int port) {
		this.port = port;
	}
	public void setHost(String host) {
		this.host = host;
	}
	
	
	protected final StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

	/* entry function for the Thread class */
	public void run() {
		globalCount();
	}
	
	/*print the content of the stream every 5 seconds*/
	protected void windowCount() {
		DataStream<String> text = env.socketTextStream(host, port, "\n");
		// parse the data, group it, window it, and aggregate the counts
		SingleOutputStreamOperator<Tuple2<String,Long>> processed = text.flatMap(new SimpleMapper())
				.keyBy(0)
				.timeWindow(Time.seconds(5))
				.reduce(new SimpleReducer());
		
		processed.print();
		try {
			env.execute("socket streamer");
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	/* a typical flat map class, map the input to its geohash as key and 1 as value*/
	public static class SimpleMapper implements FlatMapFunction<String, Tuple2<String,Long>> {
		private static final long serialVersionUID = 1L;
		@Override
		public void flatMap(String value, Collector<Tuple2<String,Long>> out) {
			JSONObject obj = new JSONObject(value);
			out.collect(new Tuple2<String,Long>(obj.getString("geohash"),1L));
		}
	}
	/* a simple reduce class to count number of events in one location*/
	public static class SimpleReducer implements ReduceFunction<Tuple2<String,Long>> {
		private static final long serialVersionUID = 1L;
		@Override
		public Tuple2<String,Long> reduce(Tuple2<String,Long> a, Tuple2<String,Long> b) {
			return new Tuple2<String,Long>(a.f0, a.f1 + b.f1);
		}
	}
	

	/* maintain a unique counter as a state to hold the number of items processed*/
	protected void globalCount() {
		DataStream<String> text = env.socketTextStream(host, port, "\n");
		DataStream<Tuple2<String, Long>> processed = text.map(new EventMapper())
					.keyBy(e->e.geohash)
					.process(new ItemCounter());
		
		processed.print();
		try {
			env.execute("socket streamer");
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	/* a simple mapper to map the string to Event */
	public static class EventMapper extends RichMapFunction<String, Event> {
		private static final long serialVersionUID = 1L;
		@Override
		public Event map(String arg0) throws Exception {
			Event e = new Event(arg0);
			e.id = "id";
			return e;
		}
	}
	/* maintain the global counter */
	public static class ItemCounter extends KeyedProcessFunction<String, Event, Tuple2<String, Long>> {
		private static final long serialVersionUID = 1L;
		ValueState<Long> total;

        @Override
        public void open(Configuration conf) {
            // register state handle
            total = getRuntimeContext().getState(
                    new ValueStateDescriptor<>("total", Types.LONG));
        }

        @Override
        public void processElement(
        		Event val,
                Context ctx,
                Collector<Tuple2<String, Long>> out) throws Exception {

            // look up start time of the last shift
            if(total.value()==null) {
            	total.update(0L);
            }
            Long curval = total.value();            
            total.update(++curval);
            if (curval%1000==0) {
                out.collect(Tuple2.of(ctx.getCurrentKey(),curval));
            }
        }
    }
	
}
