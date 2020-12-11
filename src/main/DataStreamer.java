package main;

public class DataStreamer {
    /*
     * A simple Java server to create data stream from
     * specified file. Note that this is for testing
     * purposes only. Ideally, the data stream should
     * come from third party API, Kafka etc.
     */
    public static void main(String args[]) {
		
	//Test.test_streaming_climatedata();
	//Test.test_flink_streamer_aotdata();
	//Test.test_streaming_aotdata();
	//Test.test_geohash();

	Test.test_create_stream_aotdata();
    }
}
