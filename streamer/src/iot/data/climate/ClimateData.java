package iot.data.climate;

import java.io.File;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;

import iot.common.Event;
import iot.common.TemporalSpatialData;
import iot.tools.utils.FileBatchReader;
import iot.tools.utils.Util;

/*
 * 
 *
min and max temperature from the climate data as late as 03/09/2019
444512418 records emitted from 57999 days
from:	1851/03/21 12:00:00
to:	2019/03/09 12:00:00
record with minimum temprature: 
station:	USC00507778
coordinate:	66.8236,-150.6689
element:	TMIN
mflag:	 
qflag:	 
sflag:	0
value:	-62.2
time:	1971/01/23 12:00:00
record with maximum temprature: 
station:	USC00043603
coordinate:	36.45,-116.8667
element:	TMAX
mflag:	 
qflag:	 
sflag:	6
value:	56.7
time:	1913/07/10 12:00:00
 * 
 * */
public class ClimateData extends TemporalSpatialData{
	
	protected HashMap<String,iot.data.climate.State> states = new HashMap<>();
	protected HashMap<String,Station> stations = new HashMap<>();
	protected HashMap<String,Boolean> interested_elements = new HashMap<>();
	HashMap<Long,Long> parsed_data = new HashMap<>();
	long global_count = 0;
	Element min_elm = null;
	Element max_elm = null;
	//Initialization, load the states and stations information
	public ClimateData(String meta_dir_path) {
		initialize(meta_dir_path);
	}
	
	public ClimateData() {
		
	}
	
	public void setInterestedElements(ArrayList<String> elements) {
		for(String e:elements) {
			interested_elements.put(e, Boolean.TRUE);
		}
	}
	
	@Override
	protected void emit(Event e) {
		Element elm = (Element)e;
		if(elm.element.contentEquals("TMIN")||elm.element.contentEquals("TMAX")) {
			elm.value /= 10;
		}else {
			return;
		}
		
		// update the statistics
		global_count++;
		if(!parsed_data.containsKey(e.timestamp)) {
			Long val = 0L;
			parsed_data.put(e.timestamp, val);
		}
		Long old_val = parsed_data.get(e.timestamp);
		parsed_data.put(e.timestamp, old_val+1);
		
		// update the min and max value
		if(min_elm==null||max_elm==null) {
			min_elm = elm;
			max_elm = elm;
			return;
		}
		if(elm.element.contentEquals("TMIN")&&elm.value<min_elm.value) {
			min_elm = elm;
		}
		if(elm.element.contentEquals("TMAX")&&elm.value>max_elm.value) {
			max_elm = elm;
		}
		
		// print to socket
		if(out!=null) {
			out.println(e.toString());
		}else {
			System.out.println(e.toString());
		}
		
	}
	
	@Override
	public void finalize() {
		// print the statistics of the parsed dataset
		System.out.println(global_count+" records emitted from "+parsed_data.size()+" days");
		long min = Long.MAX_VALUE;
		long max = Long.MIN_VALUE;
		Iterator<Entry<Long, Long>> it = parsed_data.entrySet().iterator();
	    while (it.hasNext()) {
	        Map.Entry<Long, Long> pair = (Map.Entry<Long, Long>)it.next();
	        if(min>pair.getKey()) {
	        	min = pair.getKey();
	        }
	        if(max<pair.getKey()) {
	        	max = pair.getKey();
	        }
	    }
	    System.out.println("from:\t"+Util.formatTimestamp(min));
	    System.out.println("to:\t"+Util.formatTimestamp(max));
	    if(min_elm!=null) {
	    	System.out.println("record with minimum temprature: ");
	    	min_elm.print();
	    	System.out.println("record with maximum temprature: ");
	    	max_elm.print();
	    }
	}
	
	@Override
	public void loadFromFiles(String path) {
		File file = new File(path);
		if(!file.exists()) {
			System.out.println(path+" does not exist");
			return;
		}
		if(file.isFile()){
			String file_name = file.getName();
			if(file_name.endsWith(".dly")) {
				String stid = file_name.substring(0,11);
				//process only the files for US stations which has meta information
				if(!stid.startsWith("US")) {
					System.out.println("station "+stid+" is not a station located in the US");
					return;
				}
				if(!stations.containsKey(stid)) {
					System.out.println("station "+stid+" has no meta data and will be passed");
					return;
				}
				
				//loading data
				FileBatchReader reader = new FileBatchReader(path, false);
				while(!reader.eof) {
					for(String line:reader.lines) {
						int year = Integer.parseInt(line.substring(11,15));
						int month = Integer.parseInt(line.substring(15,17));
						String element = line.substring(17,21);
						// this element is not interested
						if(!interested_elements.isEmpty()&&!interested_elements.containsKey(element)) {
							continue;
						}
						//traverse all 31 days in one month, each one takes 8 characters
						for(int i=0;i<31;i++) {
							Element e = new Element(year, month, i+1, element, stations.get(stid), line.substring(21+i*8,29+i*8));
							if(e.valid) {
								emit(e);
							}
						}
					}
					reader.nextBatch();
				}
				reader.closeFile();				
			}else if(file_name.endsWith(".evt")) {
				// formated file, all events for one day are stored together
				String time_str = file_name.replace(".evt", "");
				char fake_flags[] =  {' ',' ', ' '};
				try {
					long timestamp = Util.getTimestamp("yyyy-MM-dd-hh-mm-ss",time_str);
					FileBatchReader reader = new FileBatchReader(path, false);
					while(!reader.eof) {
						for(String line:reader.lines) {
							String data[] = line.split(",");
							String stid = data[0];
							String element = data[1];
							String value_str = data[2];
							System.out.println("ever been here"+stid+element+value_str+line);

							if(!stations.containsKey(stid)) {
								continue;
							}

							Element e = new Element(timestamp, element, stations.get(stid), 
									(double)Integer.parseInt(value_str), fake_flags);
							emit(e);
						}
						reader.nextBatch();
					}
					reader.closeFile();
				} catch (ParseException e) {
					e.printStackTrace();
					System.err.println(time_str+" is not a valid timestamp, should be in format "
							+ "yyyy-MM-dd-hh-mm-ss");
				}
			}
			
		}else {//is a folder
			for(File f:file.listFiles()) {
				loadFromFiles(f.getAbsolutePath());
			}
		}
	}

	@Override
	public void initialize(String path) {
		FileBatchReader reader;
		//loading states
		reader = new FileBatchReader(path+"/states.txt", false);
		while(!reader.eof) {
			for(String line:reader.lines) {
				iot.data.climate.State s = new iot.data.climate.State(line.split(" "));
				states.put(s.abbriev,s);
			}
			reader.nextBatch();
		}
		reader.closeFile();
		
		
		//loading stations
		reader = new FileBatchReader(path+"/stations.txt", false);
		while(!reader.eof) {
			for(String line:reader.lines) {
				if(line.startsWith("US")) {//we only need the data of the US states
					String st = line.substring(38,40);
					if(!st.contentEquals("  ")&&states.containsKey(st))// a valid US state
					{
						Station s = new Station(line);
						stations.put(s.ID, s);
					}
				}
				
			}
			reader.nextBatch();
		}
		reader.closeFile();
		System.out.println(states.size()+" states and "+stations.size()+" stations are loaded");
	}

}
