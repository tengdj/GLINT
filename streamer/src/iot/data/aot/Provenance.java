package iot.data.aot;

import java.text.ParseException;

import iot.tools.utils.Util;

public class Provenance {

	public int data_format_version;
	public String project_id;
	public long data_start_date;
	public long data_end_date;
	public long creation_date;
	public String url;
	
	public Provenance(String data[]) throws ParseException{
		data_format_version = Integer.parseInt(data[0]);
		project_id = data[1];
		data_start_date = Util.getTimestamp(data[2]);
		data_end_date = Util.getTimestamp(data[3]);
		creation_date = Util.getTimestamp(data[4]);
		url = data[5];
	}
	
}
