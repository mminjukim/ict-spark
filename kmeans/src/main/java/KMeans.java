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

public final class KMeans {

	static class Point implements java.io.Serializable {
		double x;
		double y;
		
		public Point() {
			this(0, 0);
		}

		public Point(double x, double y) {
			this.x = x;
			this.y = y;
		}

		double distance(Point other) {
			double dist = (x - other.x) * (x - other.x) + (y - other.y) * (y - other.y);
			return Math.sqrt(dist);
		}

		public String toString() {
			return "x=" + x + ",y=" + y;
		}
	}

	public static void main(String[] args) throws Exception {
		if (args.length < 2) {
			System.err.println("Usage: KMeans <datafile> <output-folder>");
			System.exit(1);
		}

		SparkSession spark = SparkSession.builder().appName("KMeans").getOrCreate();
		JavaSparkContext jsc = JavaSparkContext.fromSparkContext(spark.sparkContext());

		// number of clusters 
		int k = 2;

		// initialize random centers 
		Point[] centers = new Point[k];
		centers[0] = new Point(2, 2);
		centers[1] = new Point(6, 6);

		JavaRDD<String> lines = spark.read().textFile(args[0]).javaRDD();

		JavaRDD<Point> points = lines.map(s -> {
			String[] val = s.split(",");
			Point p = new Point( Double.parseDouble(val[0]), Double.parseDouble(val[1]) );
			return p;
		});

		JavaPairRDD<Integer, Point> meanPoints = null;

		for (int times = 0; times < 2; times++) {
			// broadcast centers 
			Broadcast<Point[]> broadcastCenters = jsc.broadcast(centers);

			// make rdds like ( centerId, (1,(x,y)) ) 
			JavaPairRDD<Integer, Tuple2<Integer,Point>> clustered = points.mapToPair(p -> {
				Point[] centerValue = broadcastCenters.value();
				int idx = 0;
				double minDist = p.distance(centerValue[0]);
				// calculate minimum dist among the distances between centers and p
				for (int i = 1; i < k; i++) {
					double dist = p.distance(centerValue[i]);
					if (dist < minDist) {
						idx = i;
						minDist = dist;
					}
				}
				return new Tuple2(idx, new Tuple2(1, p));
			});

			// aggregate values
			JavaPairRDD<Integer, Tuple2<Integer, Point>> aggValues = clustered.reduceByKey( (p1, p2) -> {
				int count = p1._1 + p2._1;
				Point totalPoint = new Point();
				totalPoint.x = p1._2.x + p2._2.x;
				totalPoint.y = p1._2.y + p2._2.y;
				return new Tuple2(count, totalPoint);
			});

			// calculate mean (new center) 
			meanPoints = aggValues.mapValues(tp -> {
				Point p = new Point();
				// sum of x[y] / count of points in a cluster
				p.x = tp._2.x / tp._1;
				p.y = tp._2.y / tp._1;
				return p;
			});

			List<Tuple2<Integer, Point>> newCenter = meanPoints.collect();

			// make new center to be broadcasted 
			for (int i = 0; i < newCenter.size(); i++) {
				Tuple2<Integer, Point> center = newCenter.get(i);
				int centerId = center._1;
				centers[centerId].x = center._2.x;
				centers[centerId].y = center._2.y;
				System.out.println(center);
			}

			broadcastCenters.destroy();
		}
	
		meanPoints.saveAsTextFile(args[args.length - 1]);
		spark.stop();
	}
}
