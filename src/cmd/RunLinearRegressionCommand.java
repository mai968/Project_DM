package cmd;

import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class RunLinearRegressionCommand implements Command {

    public void exec() {
        try {
            // Load the training and test datasets
            DataSource trainSource = new DataSource("data/Num80_90.arff");
            Instances trainDataset = trainSource.getDataSet();
            trainDataset.setClassIndex(trainDataset.numAttributes() - 1);

            DataSource testSource = new DataSource("data/Num20_90.arff");
            Instances testDataset = testSource.getDataSet();
            testDataset.setClassIndex(testDataset.numAttributes() - 1);

            // Create and configure the Linear Regression model
            LinearRegression model = new LinearRegression();
            model.setOptions(new String[] { "-S", "0", "-R", "1.0E-8", "-additional-stats", "-num-decimal-places", "4" });

            // Training the model and measure time to build the model
            long trainStart = System.nanoTime();
            model.buildClassifier(trainDataset);
            long trainEnd = System.nanoTime();
            double trainingTime = (trainEnd - trainStart) / 1e9;

            // Evaluate the model and measure the evaluation time.
            Evaluation eval = new Evaluation(trainDataset);
            long testStart = System.nanoTime();
            eval.evaluateModel(model, testDataset);
            long testEnd = System.nanoTime();
            double testingTime = (testEnd - testStart) / 1e9;

            // Print the regression model and analysis
            System.out.println("\n=== Linear Regression Model ===");
            System.out.println(model);

            System.out.printf("\nTime taken to build model: %.2f seconds\n", trainingTime );
            System.out.println("\n=== Evaluation on test set ===");
            System.out.println("Time taken to test model on supplied test set: " + testingTime + " seconds");

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
