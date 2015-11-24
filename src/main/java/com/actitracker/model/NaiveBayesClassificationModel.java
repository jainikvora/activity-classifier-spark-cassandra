package com.actitracker.model;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import org.apache.spark.mllib.regression.LabeledPoint;
import scala.Tuple2;

import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.api.java.function.Function;

/**
 * Created by jvalantpatel on 11/22/15.
 */
public class NaiveBayesClassificationModel {

    JavaRDD<LabeledPoint> trainingData;
    JavaRDD<LabeledPoint> testData;

    public NaiveBayesClassificationModel(JavaRDD<LabeledPoint> trainingData, JavaRDD<LabeledPoint> testData) {
        this.trainingData = trainingData;
        this.testData = testData;
    }

    public Double createModel(JavaSparkContext sc) {
        // create model
        final NaiveBayesModel model = NaiveBayes.train(this.trainingData.rdd(), 1.0);
        model.save(sc.sc(), "actitracker/naive_bayes/");

        // Evaluate model on training instances and compute training error
        JavaPairRDD<Double, Double> predictionAndLabel =
                this.testData.mapToPair(new PairFunction<LabeledPoint, Double, Double>() {
                    @Override public Tuple2<Double, Double> call(LabeledPoint p) {
                        return new Tuple2<Double, Double>(model.predict(p.features()), p.label());
                    }
                });

       // JavaPairRDD<Double, Double> predictionAndLabel = testData.mapToPair(p -> new Tuple2<>(model.predict(p.features()), p.label()));

        //Double testErrDT = 1.0 * predictionAndLabel.filter(pl -> !pl._1().equals(pl._2())).count() / testData.count();
        double accuracy = predictionAndLabel.filter(new Function<Tuple2<Double, Double>, Boolean>() {
            @Override public Boolean call(Tuple2<Double, Double> pl) {
                return pl._1().equals(pl._2());
            }
        }).count() / (double) this.testData.count();

        return accuracy;
    }
}
