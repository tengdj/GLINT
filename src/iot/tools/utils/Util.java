package iot.tools.utils;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.sql.Date;
import java.util.ArrayList;
import java.util.Locale;
import java.util.concurrent.TimeUnit;
import java.sql.Timestamp;

public class Util {
	
	public static long getTimestamp(String format, String time) throws ParseException {
		
		SimpleDateFormat dateFormat = new SimpleDateFormat(format,Locale.US);
		java.util.Date parsedDate;
		parsedDate = dateFormat.parse(time);
		java.sql.Timestamp timestamp = new Timestamp(parsedDate.getTime());
		return timestamp.getTime();	
		
	}
	
	public static String formatTimestamp(String format, long timestamp) {
		Timestamp ts=new Timestamp(timestamp);  
        Date date=new Date(ts.getTime());  
        SimpleDateFormat formater = new SimpleDateFormat(format);
        return formater.format(date);
	}
	
	public static String formatTimestamp(long timestamp){
		return Util.formatTimestamp("yyyy/MM/dd hh:mm:ss",timestamp);
	}
	
	public static long getTimestamp(String time) throws ParseException {
		return Util.getTimestamp("yyyy/MM/dd hh:mm:ss",time);
	}
	
	public static void dumpToFile(String path, String content) {
		FileWriter fw;
		try {
			fw = new FileWriter(path);
			fw.append(content);
			fw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}

	}

	public static double dist_sqr(double x, double y, double x1, double y1) {
	    double xx = x1 - x;
	    double yy = y1 - y;

	    return (double) Math.sqrt(xx * xx + yy * yy);
	}
	
	
	public static double min_distance_point_to_segment(double x, double y, 
													   double x1, double y1, 
													   double x2, double y2) {
		
		//the segment is vertical
		if(x1==x2) {
			if(y>Math.max(y1, y2)) {
				return Math.sqrt((x-x1)*(x-x1)+(y-Math.max(y1, y2))*(y-Math.max(y1, y2)));
			}else if(y<Math.min(y1, y2)){
				return Math.sqrt((x-x1)*(x-x1)+(Math.min(y1, y2)-y)*(Math.min(y1, y2)-y));
			}else {
				return Math.abs(x-x1);
			}
		}
		
		//the segment is horizontal
		if(y1==y2) {
			if(x>Math.max(x1, x2)) {
				return Math.sqrt((y-y1)*(y-y1)+(x-Math.max(x1, x2))*(x-Math.max(x1, x2)));
			}else if(x<Math.min(x1, x2)){
				return Math.sqrt((y-y1)*(y-y1)+(Math.min(x1, x2)-x)*(Math.min(x1, x2)-x));
			}else {
				return Math.abs(y-y1);
			}
		}
		
		
		double a = (y1-y2)/(x1-x2);
		double b = y1 - a*x1;
		double a1 = -1*(1/a);
		double b1 = y-a1*x;
		
		double nx = (b1-b)/(a-a1);
		double ny = a1*nx+b1;
		//the cross point is outside the segment
		if(nx>Math.max(x1, x2)||nx<Math.min(x1, x2)) {
			return Math.sqrt(Math.min((x-x1)*(x-x1)+(y-y1)*(y-y1), (x-x2)*(x-x2)+(y-y2)*(y-y2)));
		}else {
			return Math.sqrt((nx-x)*(nx-x)+(ny-y)*(ny-y));
		}	
		
	}
	
	// conversion between Celcius and Fahrenheit
	public static double celToFah(double cel) {
		return  (9.0/5) * cel + 32;
	}
	
	public static double fahToCel(double fah) {
		return (5.0/9) * (fah - 32);
	}
	
	// list all the files recursively inside a path
	public static ArrayList<String> listFiles(String path){
		File folder = new File(path);
		ArrayList<String> files = new ArrayList<String>();
		if(folder.isDirectory()) {
			for(File f:folder.listFiles()) {
				files.addAll(listFiles(f.getAbsolutePath()));
			}
		}else {
			files.add(folder.getAbsolutePath());
		}
		return files;
	}
	
	public static void clearFolder(File file) {
		if(file.isDirectory()) {
			for(File f:file.listFiles()) {
				clearFolder(f);
			}
			file.delete();
		}else {
			file.delete();			
		}
	}
	
	public static void sleep(long seconds) {
		try {
			TimeUnit.SECONDS.sleep(seconds);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
	}
	
}
