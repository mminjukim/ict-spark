import scala.Tuple2;

import org.apache.spark.sql.SparkSession;
import org.apache.spark.api.java.*;
import org.apache.spark.api.java.function.*;
import org.apache.spark.broadcast.*;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.ArrayList;
import java.util.Hashtable;
import java.util.HashMap;
import java.util.Map;
import java.util.Comparator;
import java.util.Collections;


public final class KNN {

	// Ascending sort comparator  
	static class TupleComparator implements Comparator<Tuple2<Double,String>>, Serializable {
		public int compare(Tuple2<Double,String> o1, Tuple2<Double,String> o2) {
			if(o2._1 == o1._1) return 0;
			else if(o2._1 > o1._1) return -1;
			else return 1;
		}
	}

	// Sort hashtable and return arraylist 
	public static ArrayList<Map.Entry<?, Integer>> sortValue(Hashtable<?, Integer> t){
		ArrayList<Map.Entry<?, Integer>> l = new ArrayList(t.entrySet());
		Collections.sort(l, new Comparator<Map.Entry<?, Integer>>(){
			public int compare(Map.Entry<?, Integer> o1, Map.Entry<?, Integer> o2) {
				return o1.getValue().compareTo(o2.getValue());
			}});
		return l;
	}


	static class Point implements java.io.Serializable{
		double mileage;
		double videoGameRate;
		double iceCream;
		String label;

		// Constructors 
		public Point() {
			this(0,0,0,"");
		}
		public Point(double mileage, double videoGameRate, double iceCream, String label) {
			this.mileage = mileage;
			this.videoGameRate = videoGameRate;
			this.iceCream = iceCream;
			this.label = label;
		}

		// Calculate distance 
		double distance(Point other) {
			double dist = (mileage-other.mileage)* (mileage-other.mileage);
			dist = dist + (videoGameRate-other.videoGameRate)*(videoGameRate-other.videoGameRate);
			dist = dist + (iceCream-other.iceCream)*(iceCream-other.iceCream);
			return Math.sqrt( dist );
		}

		public String toString() {
			return "mileage="+mileage+",videoGameRate="+videoGameRate+",iceCream="+iceCream + ",label="+label;
		}

	}

	public static void main(String[] args) throws Exception {
		if (args.length < 1) {
			System.err.println("Usage: KNN <file>");
			System.exit(1);
		}

		SparkSession spark = SparkSession.builder().appName("KNN").getOrCreate();
		JavaSparkContext jsc = JavaSparkContext.fromSparkContext(spark.sparkContext());

		// The number of neighbors
		int k = 5;

		// New data 
		Point query = new Point(40900, 0.3, 0.9, "");

		// Read data 
		JavaRDD<String> lines = spark.read().textFile(args[0]).javaRDD();

		// Make PointRDD (mileage, videoGameRate, iceCream, label)
		JavaRDD<Point> pointRDD = lines.map(s -> {
			String[] val = s.split("\t");
			Point p = new Point();
			p.mileage = Double.parseDouble(val[0]);
			p.videoGameRate = Double.parseDouble(val[1]);
			p.iceCream = Double.parseDouble(val[2]);
			p.label = val[3];
			return p;
		});

		// Calculate min point (to normalize)
		Point min = pointRDD.reduce((p1, p2) -> {
			Point p = new Point();
			p.mileage = Math.min( p1.mileage, p2.mileage );
			p.videoGameRate = Math.min( p1.videoGameRate, p2.videoGameRate );
			p.iceCream = Math.min( p1.iceCream, p2.iceCream );
			return p;
		});

		// Calculate max point (to normalize) 
		Point max = pointRDD.reduce((p1, p2) -> {
			Point p = new Point();
			p.mileage = Math.max( p1.mileage, p2.mileage );
			p.videoGameRate = Math.max( p1.videoGameRate, p2.videoGameRate );
			p.iceCream = Math.max( p1.iceCream, p2.iceCream );
			return p;
		});

		// Broadcast min, max point 
		Broadcast<Point> broadcastMin = jsc.broadcast(min);
		Broadcast<Point> broadcastMax = jsc.broadcast(max);

		// Normalize pointRDD 
		JavaRDD<Point> normPointRDD = pointRDD.map(p -> {
			Point minPoint = broadcastMin.value();
			Point maxPoint = broadcastMax.value();
			p.mileage = (p.mileage - minPoint.mileage) / (maxPoint.mileage - minPoint.mileage);
			p.videoGameRate = (p.videoGameRate - minPoint.videoGameRate) / (maxPoint.videoGameRate - minPoint.videoGameRate);
			p.iceCream = (p.iceCream - minPoint.iceCream) / (maxPoint.iceCream - minPoint.iceCream);
			return p;
		});

		// Normalize query point 
		query.mileage = (query.mileage - min.mileage) / (max.mileage - min.mileage);
		query.videoGameRate = (query.videoGameRate - min.videoGameRate) / (max.videoGameRate - min.videoGameRate);
		query.iceCream = (query.iceCream - min.iceCream) / (max.iceCream - min.iceCream);

		// Broadcast query point 
		Broadcast<Point> broadcastQueryPoint = jsc.broadcast(query); 

		// Calculate distances RDD (distance, label)
		JavaPairRDD<Double, String> distanceRDD = normPointRDD.mapToPair(p -> {
			Point queryPoint = broadcastQueryPoint.value();
			double dist = p.distance(queryPoint);
			return new Tuple2(dist, p.label);
		});

		// Take top k neighbors 
		List<Tuple2<Double, String>> topK = distanceRDD.takeOrdered(k, new TupleComparator());

		// Calculate label count (label, count)
		Hashtable<String, Integer> labelTable = new Hashtable<>();
		for(int i = 0; i < topK.size(); i++) { 
			Tuple2<Double, String> t = topK.get(i);
			int val = labelTable.getOrDefault(t._2, 0) + 1;
			labelTable.put(t._2, val);
		}

		// Sort in order of most labels 
		ArrayList<Map.Entry<?, Integer>> sortedLableList = sortValue(labelTable);

		System.out.println("\n\n\n\n\nStart printing...\n");
		for(int i = 0; i < sortedLableList.size(); i++) { 
			System.out.println( "   === " + sortedLableList.get(i).getKey() + " " + sortedLableList.get(i).getValue() );
		}
		System.out.println("\n\n\n\n");

		broadcastMin.destroy();
		broadcastMax.destroy();
		broadcastQueryPoint.destroy();

		spark.stop();
	}
}
