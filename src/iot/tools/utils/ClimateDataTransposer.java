package iot.tools.utils;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Stack;

import org.apache.flink.api.java.tuple.Tuple2;

import iot.common.Event;
import iot.data.climate.ClimateData;
import iot.data.climate.Element;

public class ClimateDataTransposer extends ClimateData {
	
	// only those three fields need be stored in files
	// timestamp can be given by the key, or file name
	private class TinyElement{
		public String stationid;
		public String element;
		public double value;	
		public TinyElement(Element e) {
			this.stationid = e.stationid;
			this.element = e.element;
			this.value = e.value;
		}
	}
	
	// events happened in a time period
	HashMap<Long, ArrayList<TinyElement>> time_bin = new HashMap<>();
	// note that we should not use buffered writer, it will
	// take too many spaces 
	HashMap<Long, FileWriter> writers = null;
	String output_dir = null;
	Stack<String> file_list = null;
	Tuple2<Integer, Integer> stats;
	long event_counter = 0;
	long max_event_counter = 1000000;
	int thread_id = 0;
	int total_files = -1;
	int cur_event_counter = 0;
	// internal use only
	
	public ClimateDataTransposer(int thread_id, String meta_dir_path, String out_dir, Stack<String> file_list, 
			HashMap<Long, FileWriter> writers, long buffer_size, Tuple2<Integer, Integer> stats) {
		super(meta_dir_path);
		output_dir = out_dir;
		this.file_list = file_list;
		this.thread_id = thread_id;
		this.writers = writers;
		this.max_event_counter = buffer_size;
		this.stats = stats;
		System.out.println("thread_id "+thread_id+" is initialized");
		total_files = file_list.size();
	}
	
	public ClimateDataTransposer(int thread_id, ClimateDataTransposer from) {
		this.output_dir = from.output_dir;
		this.file_list = from.file_list;
		this.thread_id = thread_id;
		this.writers = from.writers;
		this.max_event_counter = from.max_event_counter;
		this.interested_elements = from.interested_elements;
		this.states = from.states;
		this.stations = from.stations;
		this.stats = from.stats;
		System.out.println("thread_id "+thread_id+" is initialized");
		this.total_files = from.total_files;
	}
	
	public boolean isReady() {
		return total_files!=-1;
	}
	
	@Override
	public void finalize() {
		flushToFolder();
	}
	
	void flushToFolder(){
		if(event_counter==0||output_dir==null) {
			return;
		}
		
		// compose the output string 
		System.out.println("thread "+thread_id+":\tflushing to "+output_dir);
		for (HashMap.Entry<Long,ArrayList<TinyElement>> entry : time_bin.entrySet()) {

			long timestamp = entry.getKey();
			ArrayList<TinyElement> elements = entry.getValue();
			String out_str = "";
			// generate the output string for this file in a csv format
			for(TinyElement e:elements) {
				out_str += e.stationid+","+e.element+","+((int)e.value)+"\n";
			}
			elements.clear();
			
			FileWriter writer = null;
			// get the writter to the target file
			synchronized(writers) {
				// writer already been created
				if(writers.containsKey(timestamp)) {
					writer = writers.get(timestamp);
				}else {
					String filename = Paths.get(output_dir, 
							Util.formatTimestamp("yyyy-MM-dd-hh-mm-ss", timestamp)+".evt").toString();
					File outfile = new File(filename);
					if(!outfile.exists()) {
						try {
							outfile.createNewFile();
						} catch (IOException e) {
							e.printStackTrace();
							System.exit(0);
						}
					}
					try {
						writer = new FileWriter(outfile);
						writers.put(timestamp, writer);
					} catch (IOException e) {
						e.printStackTrace();
					}
				}
			}
			//append to the target file, protected by lock
			synchronized(writer) {
				try {
					writer.write(out_str);
					writer.flush();
				} catch (IOException e1) {
					// TODO Auto-generated catch block
					e1.printStackTrace();
				}
			}
		}
		time_bin.clear();
		event_counter = 0;
	}
	
	@Override
	protected void emit(Event e) {
		//push event to the bin of the certain time
		ArrayList<TinyElement> list;
		if(!time_bin.containsKey(e.timestamp)) {
			list = new ArrayList<TinyElement>();
		}else {
			list = time_bin.get(e.timestamp);
		}
		Element elm = (Element)e;
		TinyElement telm = new TinyElement(elm);
		list.add(telm);
		time_bin.put(e.timestamp, list);
		if(event_counter++>=max_event_counter) {
			flushToFolder();
		}
		cur_event_counter++;
	}

	@Override
	public void run() {
		if(file_list == null) {
			System.err.println("file list is null, please specify before using");
			finalize();
		}
		if(total_files==-1) {
			total_files = file_list.size();
		}
		while(!file_list.isEmpty()) {
			String path = null;
			synchronized(file_list) {
				if(!file_list.empty()) {
					path = file_list.pop();
					System.out.println("thread "+thread_id+":\tprocessing "+path+
							" ("+file_list.size()+" remains, "+(total_files-file_list.size())+" processed)");
				}
			}
			if(path==null) {
				break;
			}
			loadFromFiles(path);
			if(cur_event_counter>0) {
				stats.f0++;
				stats.f1 += cur_event_counter;
				cur_event_counter = 0;
			}
		}
		finalize();
	}
	
	/*
	 * the wrapper function for the transpose class
	 * */
	public static void transposeClimateData(String output_dir, String origin_dir, 
			String meta_dir, int thread_num, long buffer_size) {
		long start = System.currentTimeMillis();
		System.out.println("clearing "+output_dir);
		Util.clearFolder(new File(output_dir));
		File folder = new File(output_dir);
		while(folder.exists());
		folder.mkdirs();
		ArrayList<String> list = Util.listFiles(origin_dir);
		Stack<String> stack = new Stack<>();
		for(String s:list) {
			stack.push(s);
		}
		ArrayList<String> elements = new ArrayList<String>();
		elements.add("TMAX");
		elements.add("TMIN");
		ArrayList<ClimateDataTransposer> transposers = new ArrayList<>();
		HashMap<Long, FileWriter> writers = new HashMap<>();
		Tuple2<Integer, Integer> stats = new Tuple2<Integer, Integer>();
		stats.f0 = 0;
		stats.f1 = 0;
		//initialize threads
		ClimateDataTransposer transposer = new ClimateDataTransposer(0, meta_dir, output_dir, 
				stack, writers, buffer_size, stats);
		transposer.setInterestedElements(elements);
		transposers.add(transposer);
		for(int i=1;i<thread_num;i++) {
			//copy from the initialized object, avoid duplicate initialization
			transposers.add(new ClimateDataTransposer(i, transposer));
		}
		
		//waiting for the initialization of the threads
		boolean isready = false;
		while(!isready) {
			isready = true;
			for(ClimateDataTransposer t:transposers) {
				isready &= t.isReady();
				if(!isready) {
					break;
				}
			}
		}
		
		//start threads
		for(ClimateDataTransposer t:transposers) {
			t.start();
		}
		
		//wait for completion of threads
		for(int i=0;i<thread_num;i++) {
			try {
				transposers.get(i).join();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		System.out.println(stats.f1+" events are generated from "+stats.f0+" stations in "+writers.size()+" days");
		System.out.println("closing the writers, may take minutes");
		for (HashMap.Entry<Long, FileWriter> entry : writers.entrySet()) {
			try {
				entry.getValue().close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		writers.clear();
		long end = System.currentTimeMillis();
		System.out.println("takes "+((end-start)/1000.0)+" seconds");
	}
}
