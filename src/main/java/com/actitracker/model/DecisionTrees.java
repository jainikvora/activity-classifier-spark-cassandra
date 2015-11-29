package com.actitracker.model;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import scala.Tuple2;

import java.util.HashMap;
import java.util.Map;

public class DecisionTrees {

    JavaRDD<LabeledPoint> trainingData;
    JavaRDD<LabeledPoint> testData;


    public DecisionTrees(JavaRDD<LabeledPoint> trainingData, JavaRDD<LabeledPoint> testData) {
        this.trainingData = trainingData;
        this.testData = testData;
    }

    public Double createModel(JavaSparkContext sc) {
        // parameters
        Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
        int numClasses = 6;
        String impurity = "gini";
        int maxDepth = 9;
        int maxBins = 32; // atleast maximum number of categories M for given feature sets
        printModel(impurity, maxDepth, maxBins);
        // create model
        final DecisionTreeModel model = DecisionTree.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins);

        model.save(sc.sc(), "actitracker/decision_tree/");

        // Evaluate model on training instances and compute training error
        JavaPairRDD<Double, Double> predictionAndLabel = testData.mapToPair(p -> new Tuple2<>(model.predict(p.features()), p.label()));

        Double testErrDT = 1.0 * predictionAndLabel.filter(pl -> !pl._1().equals(pl._2())).count() / testData.count();

        return testErrDT;
    }

    public Double createModel2(JavaSparkContext sc) {
        // parameters
        Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
        int numClasses = 6;
        String impurity = "gini";
        int maxDepth = 4;
        int maxBins = 100; // atleast maximum number of categories M for given feature sets
        printModel(impurity, maxDepth, maxBins);
        // create model
        final DecisionTreeModel model = DecisionTree.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins);

        model.save(sc.sc(), "actitracker/decision_tree_2/");

        // Evaluate model on training instances and compute training error
        JavaPairRDD<Double, Double> predictionAndLabel = testData.mapToPair(p -> new Tuple2<>(model.predict(p.features()), p.label()));

        Double testErrDT = 1.0 * predictionAndLabel.filter(pl -> !pl._1().equals(pl._2())).count() / testData.count();

        return testErrDT;
    }

    public Double createModel3(JavaSparkContext sc) {
        // parameters
        Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
        int numClasses = 6;
        String impurity = "entropy";
        int maxDepth = 9;
        int maxBins = 32; // atleast maximum number of categories M for given feature sets
        printModel(impurity, maxDepth, maxBins);
        // create model
        final DecisionTreeModel model = DecisionTree.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins);

        model.save(sc.sc(), "actitracker/decision_tree_3/");

        // Evaluate model on training instances and compute training error
        JavaPairRDD<Double, Double> predictionAndLabel = testData.mapToPair(p -> new Tuple2<>(model.predict(p.features()), p.label()));

        Double testErrDT = 1.0 * predictionAndLabel.filter(pl -> !pl._1().equals(pl._2())).count() / testData.count();

        return testErrDT;
    }

    public Double createModel4(JavaSparkContext sc) {
        // parameters
        Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
        int numClasses = 6;
        String impurity = "entropy";
        int maxDepth = 4;
        int maxBins = 100; // atleast maximum number of categories M for given feature sets
        printModel(impurity, maxDepth, maxBins);
        // create model
        final DecisionTreeModel model = DecisionTree.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins);

        model.save(sc.sc(), "actitracker/decision_tree_4/");

        // Evaluate model on training instances and compute training error
        JavaPairRDD<Double, Double> predictionAndLabel = testData.mapToPair(p -> new Tuple2<>(model.predict(p.features()), p.label()));

        Double testErrDT = 1.0 * predictionAndLabel.filter(pl -> !pl._1().equals(pl._2())).count() / testData.count();

        return testErrDT;
    }

    private void printModel(String impurity, int maxDepth, int maxBins){
        System.out.println("Model with impurity -+"+impurity+" maxDepth -"+maxDepth+" maxBins - "+maxBins);
    }
}
