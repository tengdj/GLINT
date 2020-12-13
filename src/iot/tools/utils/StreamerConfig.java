package iot.tools.utils;

/* Java imports */
import java.io.File;
import java.io.InputStream;
import java.io.FileInputStream;
import java.util.Properties;

import org.apache.commons.lang3.SystemUtils;


public class StreamerConfig {

    public static Properties config;

    public static String get(String prop){
    	if(prop.endsWith("-path")){
    		String root_folder;
    		if(SystemUtils.IS_OS_WINDOWS) {
    			root_folder = config.getProperty("windows-root-folder");
    		}else {
    			root_folder = config.getProperty("linux-root-folder");
    		}
    		return root_folder+config.getProperty(prop);
    	}else {
    		return config.getProperty(prop);
    	}
    }
    
    public static int getInt(String prop) {
    	return Integer.parseInt(config.getProperty(prop));
    }
    
    public static void reloadProperties() {
    	InputStream inputStream = null;
		try {
		    config = new Properties();
		    inputStream =
		    	StreamerConfig.class.getClassLoader().getResourceAsStream("streamer.properties");
		    if(inputStream == null) {
		    	System.out.println("streamer.properties not found in jar file, try local file system");
		    	inputStream = new FileInputStream(new File("streamer.properties"));
		    }
		    config.load(inputStream);
	
		    if (!config.stringPropertyNames().contains("aot-data-file")){
		    	config.setProperty("aot-data-file", config.getProperty("aot-data-dir") + "/" + "data.csv.gz");
		    }
		    
		    inputStream.close();
		} catch (Exception e) {
		    e.printStackTrace();
		    System.exit(-1);
		} finally{
		    try{
		    	inputStream.close();
		    }catch(Exception e){
		    	System.out.println("No stream to close");
		    }
		}
    }
    
    // initialize the singleton
    static{
		StreamerConfig.reloadProperties();
    }

}
