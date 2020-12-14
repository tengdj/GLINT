package iot.data.taxi;

import java.util.ArrayList;

import iot.common.Event;
import iot.common.TemporalSpatialData;
import iot.tools.gps.Street;
import iot.tools.utils.FileBatchReader;

public class TaxiData extends TemporalSpatialData{
	
	/* limit for test only, stop after emitting such events */
	public long limits = Long.MAX_VALUE;
	ChicagoMap map = new ChicagoMap();

	@Override
	public void initialize(String path) {
		if(path.endsWith("json")) {
			map.loadFromJson(path);
		}else if(path.endsWith("csv")){
			map.loadFromCsv(path);
		}else {
			map.loadFromFormatedData(path);
		}
	}

	
	/*
	 * taxi trip is more complicated, since it is a trajectory instead of a single events	
	 * we firstly generate a list of streets which will be go through
	 * then generate a list of CurrentPositions with the given start/end timestamp and a time gap
	 * 
	 */
	void emit(Trip t) {
		ArrayList<Street> st = map.navigate(t.start_location, t.end_location);
		if(st.size()==0) {
			return;
		}		
		ArrayList<CurrentPosition> cp = t.getCurLocations(st);
	
		for(CurrentPosition c: cp) {
			// for each 
			emit(c);
		}
	}
	
	@Override
	public void emit(Event e) {
		if(out!=null) {
			out.println(e.toJson().toString());	
		}
		limits--;
	}
	
	
	
	public TaxiData(String map_file) {
		initialize(map_file);
	}
	
	@Override
	public void loadFromFiles(String path) {
		System.out.println("loading from "+path);
		iot.tools.utils.FileBatchReader fb = new FileBatchReader(path, false);
		boolean header = true;
		while(!fb.eof) {
			for(String s:fb.lines) {
				if(limits<=0) {
					break;
				}
				if(header) {
					header = false;
					continue;
				}
				String cols[] = s.split(",");
				if(cols.length!=23) {
					continue;
				}

				try {
					Trip t = new Trip(cols);
					emit(t);
				} catch (Exception e) {
					//System.out.println(s);
					continue;
				}
			}
			
			fb.nextBatch();
			break;
			
		}
		fb.closeFile();
	}

}
