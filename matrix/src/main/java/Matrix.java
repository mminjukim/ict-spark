import scala.Tuple2;

import org.apache.spark.sql.SparkSession;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.*;
import org.apache.spark.api.java.function.*;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.ArrayList;

public final class Matrix {

	public static void main(String[] args) throws Exception {

		SparkSession spark = SparkSession.builder().appName("Matrix").getOrCreate();
		
		JavaRDD<String> m1 = spark.read().textFile(args[0]).javaRDD();
		JavaRDD<String> m2 = spark.read().textFile(args[1]).javaRDD();

		int i = Integer.parseInt(args[2]);
		int x = Integer.parseInt(args[3]);
		int j = Integer.parseInt(args[4]);

		JavaPairRDD<String, Integer> m1ele = m1.flatMapToPair(s -> {
			ArrayList<Tuple2<String, Integer>> rslt = new ArrayList<>();
			String[] parts = s.split(" ");
			String row = parts[0];
			String col = parts[1];
			int value = Integer.parseInt(parts[2]);
			for (int idx = 0; idx < j; idx++) {
				String key = row + "," + idx + "," + col;
				rslt.add(new Tuple2(key, value));
			}
			return rslt.iterator();
		});
		
		JavaPairRDD<String, Integer> m2ele = m2.flatMapToPair(s -> {
			ArrayList<Tuple2<String, Integer>> rslt = new ArrayList<>();
			String[] parts = s.split(" ");
			String row = parts[0];
			String col = parts[1];
			int value = Integer.parseInt(parts[2]);
			for (int idx = 0; idx < i; idx++) {
				String key = idx + "," + col + "," + row;
				rslt.add(new Tuple2(key, value));
			}
			return rslt.iterator();
		});

		JavaPairRDD<String, Integer> unioned = m1ele.union(m2ele);
		JavaPairRDD<String, Integer> mul = unioned.reduceByKey((a, b) -> a * b);

		JavaPairRDD<String, Integer> changeKey = mul.mapToPair(tp -> {
			String[] parts = tp._1.split(",");
			String newKey = parts[0] + "," + parts[1];
			return new Tuple2(newKey, tp._2);
		});

		JavaPairRDD<String, Integer> rst = changeKey.reduceByKey((a, b) -> a + b);

		rst.saveAsTextFile(args[args.length - 1]);
		spark.stop();
	}
}
