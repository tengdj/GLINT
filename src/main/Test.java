package main;

import java.util.ArrayList;

import org.apache.commons.lang3.SystemUtils;

import iot.common.Point;
import iot.data.taxi.ChicagoMap;
import iot.data.taxi.TaxiData;
import iot.streamers.ChicagoTaxiStreamer;
import iot.tools.geohash.GeoHash;
import iot.tools.gps.Map;
import iot.tools.gps.Street;
import iot.tools.utils.StreamerConfig;

public class Test {

	// test utility functions
	public static void geohash() {
		GeoHash gh = GeoHash.fromGeohashString("tengdejun");
		System.out.println(gh.toBase32()+" = "+gh.getBoundingBoxCenterPoint().getLatitude()+","+gh.getBoundingBoxCenterPoint().getLongitude());
		GeoHash reversed_gh = GeoHash.withCharacterPrecision(8.88888888,88.888888888, 12);
		System.out.println(reversed_gh.getBoundingBoxCenterPoint().getLatitude()+","+reversed_gh.getBoundingBoxCenterPoint().getLongitude()+" = "+reversed_gh.toBase32());
	}
	
    public static void properties() {
		System.out.println(StreamerConfig.get("taxi-data-path"));
    }
    
	public static void navigate() {
		ChicagoMap st = new ChicagoMap();
		st.loadFromFormatedData(StreamerConfig.get("formated-map-data-path"));
		ArrayList<Street> nav = st.navigate(new Point(-87.6517705068,41.9426918444), new Point(-87.6288741572,41.8920726347));
		System.out.println(Map.genGeoJson(nav).toString(1));	
		st.clear();
		nav.clear();
	}

	
	// create streamer to digest different types of data    
	public static void create_streamer_taxidata() {
		ChicagoTaxiStreamer streamer = new ChicagoTaxiStreamer();
		streamer.start();
		try {
			streamer.join();
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
	}
	

	public static void streaming_taxidata() {
		
		String mappath = StreamerConfig.get("formated-map-data-path");
		String taxipath = StreamerConfig.get("taxi-data-path");

		TaxiData td = new TaxiData(mappath);
		td.setPath(taxipath);
		if(StreamerConfig.get("data-limits")!=null) {
			td.limits = StreamerConfig.getInt("data-limits");
		}
		td.start();
		
		ChicagoTaxiStreamer st = new ChicagoTaxiStreamer();
		st.start();
		
		try {
			td.join();
			st.join();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		System.out.println("completed");

	}


	public static void load_chicagomap() {
		ChicagoMap st = new ChicagoMap();
		st.loadFromCsv(StreamerConfig.get("raw-map-data-path"));
		st.dumpTo(StreamerConfig.get("formated-map-data-path"));
		st.clear();
		st.loadFromFormatedData(StreamerConfig.get("formated-map-data-path"));
	}
	
}
