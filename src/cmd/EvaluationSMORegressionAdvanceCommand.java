package cmd;

import util.Loader;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.SMOreg;
import weka.classifiers.functions.supportVector.Puk;
import weka.core.Instances;

import java.util.Random;

public class EvaluationSMORegressionAdvanceCommand {
    public void exec() {
        Instances dataset = Loader.loadArff("data/Num10.arff");

        dataset.setClassIndex(dataset.numAttributes() - 1);

        try {
            SMOreg smo = new SMOreg();

            Puk pukKernel = new Puk();
            pukKernel.setOmega(1.0);
            pukKernel.setSigma(1.0);
            smo.setKernel(pukKernel);

            smo.setC(1.0);

            smo.buildClassifier(dataset);

            Evaluation eval = new Evaluation(dataset);
            eval.crossValidateModel(smo, dataset, 10, new Random(1));

            System.out.println("=== SMOreg Model ===\n");
            System.out.println(smo);

            System.out.println("\n=== Evaluation Results ===");
            System.out.println(eval.toSummaryString());

            System.out.println("\n=== Confusion Matrix ===");
            System.out.println("Confusion Matrix is not applicable for regression tasks.");

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