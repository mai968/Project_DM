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

            long trainStart = System.nanoTime();
            smo.buildClassifier(dataset);
            long trainEnd = System.nanoTime();
            double trainingTime = (trainEnd - trainStart) / 1e9;

            Evaluation eval = new Evaluation(dataset);
            eval.crossValidateModel(smo, dataset, 10, new Random(1));

            System.out.println("=== Evaluation SMOreg Model Advance ===\n");
            System.out.println(smo);

            System.out.println("Time taken to build model: " + trainingTime + " seconds");

            System.out.println("\n=== Evaluation The Model Using Cross-Validation ===");
            System.out.println(eval.toSummaryString());

            System.out.println("\n=== Confusion Matrix ===");
            System.out.println("Confusion Matrix is not applicable for regression tasks.");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        EvaluationSMORegressionAdvanceCommand cmd = new EvaluationSMORegressionAdvanceCommand();
        cmd.exec();
    }
}