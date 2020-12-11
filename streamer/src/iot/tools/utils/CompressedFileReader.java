package iot.tools.utils;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.InputStreamReader;

import org.apache.commons.compress.compressors.CompressorException;
import org.apache.commons.compress.compressors.CompressorInputStream;
import org.apache.commons.compress.compressors.CompressorStreamFactory;

public class CompressedFileReader {
	
	public static BufferedReader getBufferedReaderForCompressedFile(String fileIn,String compress_type) {
	    FileInputStream fin;
	    BufferedReader br2 = null;
		try {
			fin = new FileInputStream(fileIn);
			BufferedInputStream bis = new BufferedInputStream(fin);
		    CompressorInputStream input = new CompressorStreamFactory().createCompressorInputStream(compress_type,bis,true);
		    br2 = new BufferedReader(new InputStreamReader(input));
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (CompressorException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	    
	    return br2;
	}
}
