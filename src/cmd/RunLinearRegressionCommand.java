package cmd;

import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class RunLinearRegressionCommand implements Command {

    public void exec() {
        try {
            DataSource trainSource = new DataSource("data/Num80_90.arff");
            Instances trainDataset = trainSource.getDataSet();
            trainDataset.setClassIndex(trainDataset.numAttributes() - 1);

            DataSource testSource = new DataSource("data/Num20_90.arff");
            Instances testDataset = testSource.getDataSet();
            testDataset.setClassIndex(testDataset.numAttributes() - 1);

            LinearRegression model = new LinearRegression();
            model.setOptions(new String[] { "-S", "0", "-R", "1.0E-8", "-additional-stats", "-num-decimal-places", "4" });

            long startTime = System.nanoTime();
            model.buildClassifier(trainDataset);
            long endTime = System.nanoTime();
            double trainingTime = (endTime - startTime) / 1e9;

            System.out.println("\n=== Classifier model (full training set) ===");
            System.out.println(model);

            Evaluation eval = new Evaluation(trainDataset);
            eval.evaluateModel(model, testDataset);

            System.out.printf("\nTime taken to build model: %.2f seconds\n", trainingTime);
            System.out.println("\n=== Evaluation on test set ===");
            long testStartTime = System.nanoTime();
            eval.evaluateModel(model, testDataset);
            long testEndTime = System.nanoTime();
            double testTime = (testEndTime - testStartTime) / 1e9;

            System.out.println("Time taken to test model on supplied test set: " + testTime + " seconds");

            System.out.println("\n=== Summary ===");
            System.out.println("Correlation coefficient                  " + eval.correlationCoefficient());
            System.out.println("Mean absolute error                      " + eval.meanAbsoluteError());
            System.out.println("Root mean squared error                  " + eval.rootMeanSquaredError());
            System.out.println("Relative absolute error                 " + eval.relativeAbsoluteError() + " %");
            System.out.println("Root relative squared error             " + eval.rootRelativeSquaredError() + " %");
            System.out.println("Total Number of Instances               " + testDataset.numInstances());

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String args[]) {
        Command cmd = new RunLinearRegressionCommand();
        cmd.exec();
    }
}
