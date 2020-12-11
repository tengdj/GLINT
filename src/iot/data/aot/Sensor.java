package iot.data.aot;

public class Sensor {

	public String ontology;
	public String subsystem;
	public String sensor;
	public String parameter;
	public String hrf_unit;
	public Integer hrf_minval = null;
	public Integer hrf_maxval = null;
	public String datasheet;
	
	public Sensor(String data[]) {
		ontology = data[0];
		subsystem = data[1];
		sensor = data[2];
		parameter = data[3];
		hrf_unit = data[4];
		if(!data[5].isEmpty()) {
			hrf_minval = Integer.parseInt(data[5]);
		}		
		if(!data[6].isEmpty()) {
			hrf_maxval = Integer.parseInt(data[6]);
		}
		datasheet = data[7];
	}
}
