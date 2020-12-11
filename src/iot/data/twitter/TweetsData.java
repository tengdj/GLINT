package iot.data.twitter;

import java.io.BufferedReader;
import java.io.File;

import iot.common.Event;
import iot.tools.utils.CompressedFileReader;
import iot.tools.utils.FileBatchReader;

public class TweetsData extends iot.common.TemporalSpatialData{

	//emit the tweet record to stream
	@Override
	public void emit(Event t) {
		out.println(t.toString());
	}
	
	@Override
	public void loadFromFiles(String path) {
		File f = new File(path);
		if(f.isFile()) {
			//decompress and load the data inside
			if(f.getName().endsWith(".bz2")) {
				System.out.println("processing "+path);
				BufferedReader br = CompressedFileReader.getBufferedReaderForCompressedFile(path,"bzip2");
				if(br!=null) {
					FileBatchReader fr = new FileBatchReader(br, false);
					while(!fr.eof) {
						for(String s:fr.lines) {
							emit(new Tweet(s));
						}
						fr.nextBatch();
					}
				}
			}

		}else {
			for(File fs:f.listFiles()) {
				loadFromFiles(fs.getAbsolutePath());
			}
		}
	}
	
	//do some initilizing job
	public TweetsData(String folder) {
		initialize(folder);
	}

	@Override
	public void initialize(String path) {
		
	}
	
	
}
