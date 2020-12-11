package main;

import java.util.ArrayList;
import iot.common.Point;
import iot.data.aot.AOTData;
import iot.data.climate.ClimateData;
import iot.data.taxi.ChicagoMap;
import iot.data.taxi.TaxiData;
import iot.streamers.AOTStreamer;
import iot.streamers.ChicagoTaxiStreamer;
import iot.streamers.ClimateStreamer;
import iot.tools.geohash.GeoHash;
import iot.tools.gps.Map;
import iot.tools.gps.Street;
import iot.tools.utils.StreamerConfig;

public class Test {

	public static void test_geohash() {
		GeoHash gh = GeoHash.fromGeohashString("tengdejun");
		System.out.println(gh.toBase32()+" = "+gh.getBoundingBoxCenterPoint().getLatitude()+","+gh.getBoundingBoxCenterPoint().getLongitude());
		GeoHash reversed_gh = GeoHash.withCharacterPrecision(8.88888888,88.888888888, 12);
		System.out.println(reversed_gh.getBoundingBoxCenterPoint().getLatitude()+","+reversed_gh.getBoundingBoxCenterPoint().getLongitude()+" = "+reversed_gh.toBase32());
	}
	
    public static void test_properties() {
		System.out.println(StreamerConfig.get("aot-data-dir"));
		System.out.println(StreamerConfig.get("aot-data-file"));
		System.out.println(StreamerConfig.get("taxi-data-path"));
    }

    public static void test_create_stream_aotdata() {
    	AOTData at = new AOTData(StreamerConfig.get("aot-data-dir"));
    	at.setPath(StreamerConfig.get("aot-data-file"));
    	at.start();
    }
    
    public static void test_flink_streamer_aotdata() {
    	AOTStreamer as = new AOTStreamer();
    	as.start();

		try{
		    as.join();
		} catch(Exception e){
			e.printStackTrace();
		}
    }

	public static void test_streaming_taxidata() {
		TaxiData td = new TaxiData("data/chicago/formated");
		td.setPath("data/chicago/Taxi_Trips.csv");
		td.limits = 1000;
		td.start();
		
		ChicagoTaxiStreamer st = new ChicagoTaxiStreamer();
		st.start();
	}
	
	public static void test_load_aotdata() {
		AOTData dt = new AOTData("data/aotdata");
		dt.loadFromFiles("data/aotdata/data.csv");
	}
	
	public static void test_load_climatedata() {
		ClimateData cd = new ClimateData(StreamerConfig.get("climate-meta-dir"));
		ArrayList<String> elements = new ArrayList<String>();
		elements.add("TMAX");
		elements.add("TMIN");
		cd.setInterestedElements(elements);
		cd.setPath(StreamerConfig.get("climate-data-dir-tiny"));
		cd.start();
		try {
			cd.join();
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
	}
	
	public static void test_load_climatedata_local() {
		ClimateData cd = new ClimateData(StreamerConfig.get("climate-meta-dir"));
		ArrayList<String> elements = new ArrayList<String>();
		elements.add("TMAX");
		elements.add("TMIN");
		cd.setInterestedElements(elements);
		cd.loadFromFiles(StreamerConfig.get("climate-data-dir-tiny"));
	}
	
	public static void test_streaming_climatedata() {
		ClimateStreamer cs = new ClimateStreamer();
		cs.start();
		try {
			cs.join();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public static void test_load_chicagomap() {
		ChicagoMap st = new ChicagoMap();
		st.loadFromCsv("data/chicago/transportation.csv");
		st.dumpTo("data/chicago/formated");
		st.clear();
		st.loadFromFormatedData("data/chicago/formated");
	}
	
	public static void test_navigate() {
		ChicagoMap st = new ChicagoMap();
		st.loadFromFormatedData("data/chicago/formated");
		ArrayList<Street> nav = st.navigate(new Point(-87.62076287,41.89833179), new Point(-87.90303966,41.97907082));
		System.out.println(Map.genGeoJson(nav).toString(1));	
		st.clear();
		nav.clear();
	}
	
	public static void test_load_taxidata() {
		TaxiData td = new TaxiData("data/chicago/formated");
		td.loadFromFiles("data/chicago/Taxi_Trips.csv");
	}
	
	
}
