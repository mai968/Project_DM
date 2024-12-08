package cmd;

import weka.classifiers.functions.SMOreg;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.functions.supportVector.RBFKernel; // Import kernel

public class RunSMORegressionCommand implements Command {
    public void exec() {
        try {
            DataSource trainSource = new DataSource("data/Num80_90.arff");
            Instances trainData = trainSource.getDataSet();
            trainData.setClassIndex(trainData.numAttributes() - 1);

            DataSource testSource = new DataSource("data/Num20_90.arff");
            Instances testData = testSource.getDataSet();
            testData.setClassIndex(testData.numAttributes() - 1);

            SMOreg smo = new SMOreg();

            RBFKernel rbfKernel = new RBFKernel();
            rbfKernel.setGamma(0.01);
            smo.setKernel(rbfKernel);

            smo.setC(1.0);

            long startTime = System.nanoTime();
            smo.buildClassifier(trainData);
            long endTime = System.nanoTime();

            double timeTaken = (endTime - startTime) / 1e9;
            System.out.println("Time taken to build model: " + timeTaken + " seconds");

            Evaluation eval = new Evaluation(trainData);
            eval.evaluateModel(smo, testData);

            System.out.println("=== SMOreg Model ===");
            System.out.println(smo);

            System.out.println("Time taken to build model: " + timeTaken + " seconds");

            System.out.println("\n=== Evaluation Results ===");
            System.out.println(eval.toSummaryString("\nSummary:\n", false));
            System.out.println("Correlation coefficient: " + eval.correlationCoefficient());
            System.out.println("Mean Absolute Error (MAE): " + eval.meanAbsoluteError());
            System.out.println("Root Mean Squared Error (RMSE): " + eval.rootMeanSquaredError());
            System.out.println("Relative Absolute Error (%): " + eval.relativeAbsoluteError());
            System.out.println("Root Relative Squared Error (%): " + eval.rootRelativeSquaredError());

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    public static void main(String args[]) {
        Command cmd = new RunSMORegressionCommand();
        cmd.exec();
    }
}
