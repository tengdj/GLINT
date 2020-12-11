package iot.data.aot;

import java.text.ParseException;
import java.util.HashMap;

import iot.common.Point;
import iot.common.TemporalSpatialData;
import iot.tools.utils.CompressedFileReader;
import iot.tools.utils.FileBatchReader;

public class AOTData extends TemporalSpatialData{
	
	HashMap<Long, Node> nodes = new HashMap<>();
	HashMap<String, Provenance> provenances = new HashMap<>();
	HashMap<String, Sensor> sensors = new HashMap<>();
	
	//initialize nodes, provenances and sensors
	public AOTData(String path) {
		initialize(path);
	}

	@Override
	public void loadFromFiles(String path) {
		
		//loading data
		FileBatchReader reader = null;
		if(path.endsWith(".csv")) {
			reader = new FileBatchReader(path,true);
		}else if(path.endsWith(".gz")) {
			reader = new FileBatchReader(CompressedFileReader.getBufferedReaderForCompressedFile(path, "gz"), false);
		}else {
			System.err.println("unsupported file format");
		}
		while(!reader.eof) {
			for(String line:reader.lines) {
				Thing d;
				try {
					d = new Thing(line.split(","));
				} catch (ParseException e) {
					e.printStackTrace();
					continue;
				}
				//validate the datum parsed
//				if(!sensors.containsKey(d.sensor+d.parameter)) {
//					System.out.println(d.sensor+" "+d.parameter+" does not exist");
//					System.out.println(line+"\n");
//				}
				//the node_id must be valid
				if(!nodes.containsKey(d.node_id)) {
					System.out.println("node "+d.node_id+" does not exist");
				}else {
					//assign the coordinate information to one "thing"
					Node n= nodes.get(d.node_id);
					d.coordinate = new Point(n.longitude,n.latitude);
					emit(d);
				}
			}
			reader.nextBatch();
		}
		reader.closeFile();
	}

	@Override
	public void initialize(String path) {
		String filepath;
		FileBatchReader reader;
		//loading data from provenance file
		filepath = path+"/provenance.csv";
		reader = new FileBatchReader(filepath,true);
		while(!reader.eof) {
			for(String line:reader.lines) {
				Provenance p;
				try {
					p = new Provenance(line.split(","));
				} catch (ParseException e) {
					e.printStackTrace();
					continue;
				}
				provenances.put(p.project_id,p);
			}
			reader.nextBatch();
		}
		reader.closeFile();
		
		//loading sensor
		filepath = path+"/sensors.csv";
		reader = new FileBatchReader(filepath, true);
		while(!reader.eof) {
			for(String line:reader.lines) {
				Sensor s = new Sensor(line.split(","));
				sensors.put(s.sensor+s.parameter,s);
			}
			reader.nextBatch();
		}
		reader.closeFile();
		
		//loading nodes
		filepath = path+"/nodes.csv";
		reader = new FileBatchReader(filepath, true);
		while(!reader.eof) {
			for(String line:reader.lines) {
				//contains a string with ","
				String newline = "";
				if(line.contains("\"")) {
					boolean instring = false;
					for(char ch:line.toCharArray()) {
						if(ch=='"') {
							if(instring) {
								instring = false;
							}else {
								instring = true;
							}
						}
						if(ch==',') {
							if(instring) {//replace comma in string to tab
								ch = '\t';
							}
						}
						newline += ch;
					}
				}else {
					newline = line;
				}				
				Node n;
				try {
					n = new Node(newline.split(","));
				} catch (ParseException e) {
					e.printStackTrace();
					continue;
				}
				nodes.put(n.node_id,n);
				
			}
			reader.nextBatch();
		}
		reader.closeFile();
	}
}
