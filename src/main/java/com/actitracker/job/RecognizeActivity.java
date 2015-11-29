package com.actitracker.job;


import com.actitracker.data.Constants;
import com.actitracker.data.DataManager;
import com.actitracker.data.ExtractFeature;
import com.actitracker.data.PrepareData;
import com.actitracker.model.DecisionTrees;
import com.actitracker.model.MultinomialLogisticRegression;
import com.actitracker.model.RandomForests;
import com.datastax.spark.connector.japi.CassandraRow;
import com.datastax.spark.connector.japi.rdd.CassandraJavaRDD;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.log4j.Logger;
import org.apache.log4j.Level;
import java.util.*;

import static com.actitracker.data.ExtractFeature.*;
import static com.datastax.spark.connector.japi.CassandraJavaUtil.*;
import static java.util.Arrays.*;

public class RecognizeActivity {

    private static List<String> ACTIVITIES = asList("Standing", "Jogging", "Walking", "Sitting", "Upstairs", "Downstairs");

    public static void main(String[] args) {

        // define Spark context
        SparkConf sparkConf = new SparkConf()
                .setAppName("Activity classifier")
                .set("spark.cassandra.connection.host", "127.0.0.1")
                .setMaster("local[*]");

        Logger.getLogger("org").setLevel(Level.WARN);
        Logger.getLogger("akka").setLevel(Level.WARN);
        Logger log = Logger.getLogger("org");

        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        // retrieve data from Cassandra and create an CassandraRDD
        CassandraJavaRDD<CassandraRow> cassandraRowsRDD = javaFunctions(sc).cassandraTable("actitracker", "users");
        JavaRDD<Integer> users = cassandraRowsRDD.select("user_id").distinct().map(CassandraRow::toMap).map(entry -> (int) entry.get("user_id")).cache();
//
        Set<Integer> user_ids = new HashSet<>(users.collect());
        List<LabeledPoint> labeledPoints = new ArrayList<>();

        for (Integer i : user_ids) {
            for (String activity : ACTIVITIES) {
                log.debug("Processing user id: " + i + " --- for activity: " + activity);
                // create bucket of sorted data by ascending timestamp by (user, activity)
                JavaRDD<Long> times = cassandraRowsRDD.select("timestamp")
                        .where("user_id=? AND activity=?", i, activity)
                        .withAscOrder()
                        .map(CassandraRow::toMap)
                        .map(entry -> (long) entry.get("timestamp"))
                        .cache();

                JavaRDD<CassandraRow> dataTotal = cassandraRowsRDD.select("timestamp", "acc_x", "acc_y", "acc_z")
                        .where("user_id=? AND activity=?", i, activity)
                        .withAscOrder().cache();
                log.debug(">> Data row count: " + times.count());

                // if data
                if (100 < times.count()) {

                    //////////////////////////////////////////////////////////////////////////////
                    // PREPARE THE DATA: define the windows for each activity records intervals //
                    //////////////////////////////////////////////////////////////////////////////
                    List<Long[]> intervals = defineWindows(times);
                    for (Long[] interval : intervals) {
                        log.debug("Interval Start: " + interval[0] + ", Interval End: " + interval[1] + ", Number of windows: " + interval[2]);
                        for (int j = 0; j <= interval[2]; j++) {

                            JavaRDD<CassandraRow> data = getDataIntervalData(dataTotal, interval[0], j);

                            if (data.count() > 0) {
                                // transform into double array
                                JavaRDD<double[]> doubles = DataManager.toDouble(data);
                                // transform into vector without timestamp
                                JavaRDD<Vector> vectors = doubles.map(Vectors::dense);
                                // data with only timestamp and acc
                                JavaRDD<long[]> timestamp = DataManager.withTimestamp(data);

                                ////////////////////////////////////////
                                // extract features from this windows //
                                ////////////////////////////////////////
                                ExtractFeature extractFeature = new ExtractFeature(vectors);

                                // the average acceleration
                                double[] mean = extractFeature.computeAvgAcc();

                                // the variance
                                double[] variance = extractFeature.computeVariance();

                                // the average absolute difference
                                double[] avgAbsDiff = computeAvgAbsDifference(doubles, mean);

                                // the average resultant acceleration
                                double resultant = computeResultantAcc(doubles);

                                // the average time between peaks
                                double avgTimePeak = extractFeature.computeAvgTimeBetweenPeak(timestamp);

                                // Let's build LabeledPoint, the structure used in MLlib to create and a predictive model
                                LabeledPoint labeledPoint = getLabeledPoint(activity, mean, variance, avgAbsDiff, resultant, avgTimePeak);
                                labeledPoints.add(labeledPoint);
                            }
                        }
                    }
                }
            }
        }

        // ML part with the models: create model prediction and train data on it //
        if (labeledPoints.size() > 0) {

            log.debug("Creating models");
            // data ready to be used to build the model
            JavaRDD<LabeledPoint> data = sc.parallelize(labeledPoints);

            // Split data into 2 sets : training (60%) and test (40%).
            JavaRDD<LabeledPoint>[] splits = data.randomSplit(new double[]{Constants.training, Constants.test});
            JavaRDD<LabeledPoint> trainingData = splits[0].cache();
            JavaRDD<LabeledPoint> testData = splits[1];
            // With DecisionTree
            double errDT = new DecisionTrees(trainingData, testData).createModel(sc);
            // With Random Forest
            double errRF = new RandomForests(trainingData, testData).createModel(sc);
            // with logistic regression
            double errLR = new MultinomialLogisticRegression(trainingData, testData).createModel(sc);

            System.out.println("sample size " + data.count());
            System.out.println("Test Error Decision Tree: " + errDT);
            System.out.println("Test Error Random Forest: " + errRF);
            System.out.println("Test Error Logistic Regression: " + errLR);
        }
    }

    private static List<Long[]> defineWindows(JavaRDD<Long> times) {
        // first find jumps to define the continuous periods of data
        Long firstElement = times.first();
        Long lastElement = times.sortBy(time -> time, false, 1).first();

        // compute the difference between each timestamp
        JavaPairRDD<Long[], Long> tsBoundariesDiff = PrepareData.boudariesDiff(times, firstElement, lastElement);

        // define periods of recording
        // if the difference is greater than 100 000 000, it must be different periods of recording
        // ({min_boundary, max_boundary}, max_boundary - min_boundary > 100 000 000)
        JavaPairRDD<Long, Long> jumps = PrepareData.defineJump(tsBoundariesDiff);

        // Now define the intervals
        return PrepareData.defineInterval(jumps, firstElement, lastElement, Constants.interval);
    }

    /**
     * build the data set with label & features (11)
     * activity, mean_x, mean_y, mean_z, var_x, var_y, var_z, avg_abs_diff_x, avg_abs_diff_y, avg_abs_diff_z, res, peak_y
     */
    private static LabeledPoint getLabeledPoint(String activity, double[] mean, double[] variance, double[] avgAbsDiff, double resultant, double avgTimePeak) {
        // First the feature
        double[] features = new double[]{
                mean[0],
                mean[1],
                mean[2],
                variance[0],
                variance[1],
                variance[2],
                avgAbsDiff[0],
                avgAbsDiff[1],
                avgAbsDiff[2],
                resultant,
                avgTimePeak
        };

        // Now the label: by default 0 for Walking
        double label = 0;

        if ("Jogging".equals(activity)) {
            label = 1;
        } else if ("Standing".equals(activity)) {
            label = 2;
        } else if ("Sitting".equals(activity)) {
            label = 3;
        } else if ("Upstairs".equals(activity)) {
            label = 4;
        } else if ("Downstairs".equals(activity)) {
            label = 5;
        }

        return new LabeledPoint(label, Vectors.dense(features));
    }

    /**
     * Get data slices based on window interval.
     * @param data
     * @param interval
     * @param j
     * @return JavaRDD<CassandraRow> - interval data from the jump
     */
    private static JavaRDD<CassandraRow> getDataIntervalData(JavaRDD<CassandraRow> data, long interval, int j) {
        return data.filter(raw ->
                        Long.valueOf(raw.getString("timestamp")) < interval + (j + 1) * Constants.interval && Long.valueOf(raw.getString("timestamp")) > interval + j * Constants.interval
        );
    }
}
