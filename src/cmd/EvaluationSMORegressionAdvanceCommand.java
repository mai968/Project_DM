package cmd;

import util.Loader;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.SMOreg;
import weka.classifiers.functions.supportVector.Puk;
import weka.core.Instances;

import java.util.Random;

public class EvaluationSMORegressionAdvanceCommand {
    public void exec() {
        // Load the dataset
        Instances dataset = Loader.loadArff("data/Num10.arff");
        dataset.setClassIndex(dataset.numAttributes() - 1);

        try {
            // Create and configure SMOreg
            SMOreg smo = new SMOreg();

            // Configure Puk kernel (Pearson VII function-based Universal Kernel)
            Puk pukKernel = new Puk();
            pukKernel.setOmega(1.0);
            pukKernel.setSigma(1.0);
            smo.setKernel(pukKernel);

            // Configure SMOreg hyperparameters
            smo.setC(1.0);

            // Training the model and measure time to build the model
            long trainStart = System.nanoTime();
            smo.buildClassifier(dataset);
            long trainEnd = System.nanoTime();
            double trainingTime = (trainEnd - trainStart) / 1e9;


            // Evaluate the model using 10-fold cross-validation
            Evaluation eval = new Evaluation(dataset);
            eval.crossValidateModel(smo, dataset, 10, new Random(1));


            // Print the Evaluation SMOreg Model Advance
            System.out.println("=== Evaluation SMOreg Model Advance ===\n");
            System.out.println(smo);

            System.out.println("Time taken to build model: " + trainingTime + " seconds");

            // Print evaluation results
            System.out.println("\n=== Evaluation The Model Using Cross-Validation ===");
            System.out.println(eval.toSummaryString());

            // Print confusion matrix (not applicable for regression)
            System.out.println("\n=== Confusion Matrix ===");
            System.out.println("Confusion Matrix is not applicable for regression tasks.");

            // Additional metrics for regression
//            System.out.println("\n=== Regression Metrics ===");
//            System.out.println("Correlation Coefficient: " + eval.correlationCoefficient());
//            System.out.println("Mean Absolute Error (MAE): " + eval.meanAbsoluteError());
//            System.out.println("Root Mean Squared Error (RMSE): " + eval.rootMeanSquaredError());
//            System.out.println("Relative Absolute Error (%): " + eval.relativeAbsoluteError());
//            System.out.println("Root Relative Squared Error (%): " + eval.rootRelativeSquaredError());

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        EvaluationSMORegressionAdvanceCommand cmd = new EvaluationSMORegressionAdvanceCommand();
        cmd.exec();
    }
}