package cmd;

import weka.classifiers.functions.SMOreg;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.functions.supportVector.RBFKernel;

public class RunSMORegressionCommand implements Command {
    public void exec() {
        try {
            // Load the training and test datasets
            DataSource trainSource = new DataSource("data/Num80_90.arff");
            Instances trainData = trainSource.getDataSet();
            trainData.setClassIndex(trainData.numAttributes() - 1);

            DataSource testSource = new DataSource("data/Num20_90.arff");
            Instances testData = testSource.getDataSet();
            testData.setClassIndex(testData.numAttributes() - 1);

            // Create and configure the SMOreg model
            SMOreg smo = new SMOreg();

            // Kernel setup (eg RBF Kernel - Gaussian Kernel)
            RBFKernel rbfKernel = new RBFKernel();
            rbfKernel.setGamma(0.01);
            smo.setKernel(rbfKernel);

            smo.setC(1.0);

            // Training the model and measure time to build the model
            long trainStart = System.nanoTime();
            smo.buildClassifier(trainData);
            long trainEnd = System.nanoTime();
            double trainingTime = (trainEnd - trainStart) / 1e9;

            // Evaluate the model and measure the evaluation time.
            Evaluation eval = new Evaluation(trainData);
            long testStart = System.nanoTime();
            eval.evaluateModel(smo, testData);
            long testEnd = System.nanoTime();
            double testingTime = (testEnd - testStart) / 1e9;

            // Print the SMOreg and analysis
            System.out.println("=== SMOreg Model ===");
            System.out.println(smo);

            System.out.println("Time taken to build model: " + trainingTime + " seconds");
            System.out.println("\n=== Evaluation on test set ===");
            System.out.println("Time taken to test model on supplied test set: " + testingTime + " seconds");

            System.out.println("\n=== Evaluation Results ===");
            System.out.println(eval.toSummaryString("\nSummary:\n", false));
//            System.out.println("Correlation coefficient: " + eval.correlationCoefficient());
//            System.out.println("Mean Absolute Error (MAE): " + eval.meanAbsoluteError());
//            System.out.println("Root Mean Squared Error (RMSE): " + eval.rootMeanSquaredError());
//            System.out.println("Relative Absolute Error (%): " + eval.relativeAbsoluteError());
//            System.out.println("Root Relative Squared Error (%): " + eval.rootRelativeSquaredError());

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    public static void main(String args[]) {
        Command cmd = new RunSMORegressionCommand();
        cmd.exec();
    }
}
