package iot.data.climate;

import iot.tools.geohash.GeoHash;

public class Station {
	
	public String ID;
	public double longitude;
	public double latitude;
	public double elevation;
	public String state_abbriev;
	public String name;
	public String gsn;
	public String hcn_crn;
	public int wmo_id;
	public String geohash;
	
	public Station(String line) {
		
		state_abbriev = line.substring(38,40);
		assert !state_abbriev.contentEquals("  "):"the station should be in america";
		ID = line.substring(0, 11);
		latitude = Double.parseDouble(line.substring(12, 20));
		longitude = Double.parseDouble(line.substring(21, 30));
		elevation = Double.parseDouble(line.substring(31,37));
		name = line.substring(41,71);
		gsn = line.substring(72,75);
		hcn_crn = line.substring(76,79);
		if(line.length()<85||line.substring(80,line.length()).contentEquals("     ")) {
			wmo_id = -1;
		}else {
			wmo_id = Integer.parseInt(line.substring(80,85));
		}
		//hash the longitude and latitude
		geohash = GeoHash.withCharacterPrecision(latitude, longitude, 12).toBase32();
		

		//System.out.println(ID+"   "+longitude+"  "+latitude+"  "+elevation+" "+state_abbriev+" "+name+" "+wmo_id);

		
	}

}
