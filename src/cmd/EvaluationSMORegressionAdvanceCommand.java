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
        Instances dataset = Loader.loadArff("data/Num10.arff"); // Replace with your dataset path

        // Set the class index
        dataset.setClassIndex(dataset.numAttributes() - 1); // Cột cuối là nhãn cần dự đoán

        try {
            // Create and configure SMOreg
            SMOreg smo = new SMOreg();

            // Configure Puk kernel (Pearson VII function-based Universal Kernel)
            Puk pukKernel = new Puk();
            pukKernel.setOmega(1.0); // Omega mặc định
            pukKernel.setSigma(1.0); // Sigma mặc định
            smo.setKernel(pukKernel);

            // Configure SMOreg hyperparameters
            smo.setC(1.0); // Regularization parameter

            // Train the model on the entire dataset
            smo.buildClassifier(dataset);

            // Evaluate the model using cross-validation
            Evaluation eval = new Evaluation(dataset);
            eval.crossValidateModel(smo, dataset, 10, new Random(1)); // 10-fold cross-validation

            // Print the SMOreg model
            System.out.println("=== SMOreg Model ===\n");
            System.out.println(smo);

            // Print evaluation results
            System.out.println("\n=== Evaluation Results ===");
            System.out.println(eval.toSummaryString()); // Tổng quan kết quả đánh giá

            // Print confusion matrix (not applicable for regression)
            System.out.println("\n=== Confusion Matrix ===");
            System.out.println("Confusion Matrix is not applicable for regression tasks.");

            // Additional metrics for regression
            System.out.println("\n=== Regression Metrics ===");
            System.out.println("Correlation Coefficient: " + eval.correlationCoefficient());
            System.out.println("Mean Absolute Error (MAE): " + eval.meanAbsoluteError());
            System.out.println("Root Mean Squared Error (RMSE): " + eval.rootMeanSquaredError());
            System.out.println("Relative Absolute Error (%): " + eval.relativeAbsoluteError());
            System.out.println("Root Relative Squared Error (%): " + eval.rootRelativeSquaredError());

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        EvaluationSMORegressionAdvanceCommand cmd = new EvaluationSMORegressionAdvanceCommand();
        cmd.exec();
    }
}