/*
 * Copyright 2010, Silvio Heuberger @ IFS www.ifs.hsr.ch
 *
 * This code is release under the Apache License 2.0.
 * You should have received a copy of the license
 * in the LICENSE file. If you have not, see
 * http://www.apache.org/licenses/LICENSE-2.0
 */
package iot.common;

import java.io.Serializable;

/**
 * {@link Point} encapsulates coordinates on the earths surface.<br>
 * Coordinate projections might end up using this class...
 */
public class Point implements Serializable {
	private static final long serialVersionUID = 7457963026513014856L;
	public final double longitude;
	public final double latitude;

	public Point(double longitude, double latitude) {
		this.latitude = latitude;
		this.longitude = longitude;
		if (Math.abs(latitude) > 90 || Math.abs(longitude) > 180) {
			throw new IllegalArgumentException("The supplied coordinates " + this + " are out of range.");
		}
	}

	public Point(Point other) {
		this(other.longitude,other.latitude);
	}

	public double getLatitude() {
		return latitude;
	}

	public double getLongitude() {
		return longitude;
	}

	@Override
	public String toString() {
		return String.format("(" + latitude + "," + longitude + ")");
	}

	@Override
	public boolean equals(Object obj) {
		if (obj instanceof Point) {
			Point other = (Point) obj;
			return latitude == other.latitude && longitude == other.longitude;
		}
		return false;
	}

	@Override
	public int hashCode() {
		int result = 42;
		long latBits = Double.doubleToLongBits(latitude);
		long lonBits = Double.doubleToLongBits(longitude);
		result = 31 * result + (int) (latBits ^ (latBits >>> 32));
		result = 31 * result + (int) (lonBits ^ (lonBits >>> 32));
		return result;
	}
	
	public double distance(Point p) {
		if(p==null) {
			return 0;
		}
		return Math.sqrt((longitude-p.longitude)*(longitude-p.longitude)
						+(latitude-p.latitude)*(latitude-p.latitude));
	}
	
}