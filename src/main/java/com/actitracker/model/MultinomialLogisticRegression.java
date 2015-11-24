package com.actitracker.model;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import scala.Tuple2;

import java.util.HashMap;
import java.util.Map;

/**
 * Created by jvalantpatel on 11/22/15.
 */
public class MultinomialLogisticRegression {

    JavaRDD<LabeledPoint> trainingData;
    JavaRDD<LabeledPoint> testData;

    public MultinomialLogisticRegression(JavaRDD<LabeledPoint> trainingData, JavaRDD<LabeledPoint> testData) {
        this.trainingData = trainingData;
        this.testData = testData;
    }

    public Double createModel(JavaSparkContext sc) {
        // parameters
        Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
        final int numClasses = 6;


        // create model
        final LogisticRegressionModel model = new LogisticRegressionWithLBFGS()
                .setNumClasses(numClasses)
                .run(this.trainingData.rdd());

        model.save(sc.sc(), "actitracker/logistic_regression/");

        // Evaluate model on training instances and compute training error
        JavaPairRDD<Double, Double> predictionAndLabel = testData.mapToPair(p -> new Tuple2<>(model.predict(p.features()), p.label()));

//        JavaRDD<Tuple2<Object, Object>> predictionAndLabels = this.testData.map(
//                new Function<LabeledPoint, Tuple2<Object, Object>>() {
//                    public Tuple2<Object, Object> call(LabeledPoint p) {
//                        Double prediction = model.predict(p.features());
//                        return new Tuple2<Object, Object>(prediction, p.label());
//                    }
//                }
//        );

        Double testErrDT = 1.0 * predictionAndLabel.filter(pl -> !pl._1().equals(pl._2())).count() / testData.count();

        return testErrDT;
    }
}
