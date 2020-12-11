package iot.tools.utils;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

public class FileBatchReader {

	public boolean eof = false;
	public ArrayList<String> lines = new ArrayList<String>();
	public int batchLimit =200000;
	BufferedReader br;
	
	public FileBatchReader(String path, boolean skip_head) {
		try {
			br = new BufferedReader(new FileReader(path));
			//skip the head of the file
			if(skip_head) {
				br.readLine();
			}
			nextBatch();
		} catch (IOException e) {
			eof = true;
			e.printStackTrace();
		} 
	}
	
	public void setBufferSize(int buffer_size) {
		this.batchLimit = buffer_size;
	}
	
	public FileBatchReader(BufferedReader br, boolean skip_head) {
		this.br = br;
		nextBatch();
	}
	
	public String readAll() {
		String str = "";
		while(!eof) {
			for(String line:lines) {
				str += line;
			}
			nextBatch();
		}
		return str;
	}
	
	public ArrayList<String> readLines() {
		ArrayList<String> ret_lines = new ArrayList<String>();
		while(!eof) {
			ret_lines.addAll(lines);
			nextBatch();
		}
		return ret_lines;
	}
	
	public void nextBatch() {
		lines.clear();
		String line;
		int linenum = 0;
		try {
			while((line=br.readLine())!=null) {
				//System.out.println(line);
				lines.add(line);
				if(++linenum>=batchLimit) {
					break;
				}
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			eof = true;
		}
		if(linenum==0) {
			eof = true;
		}
	}
	
	public void closeFile() {
		try {
			br.close();
			eof = true;
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
}
