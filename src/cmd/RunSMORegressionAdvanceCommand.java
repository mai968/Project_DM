package cmd;

import weka.classifiers.functions.SMOreg;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.functions.supportVector.Puk;

public class RunSMORegressionAdvanceCommand implements Command {
    public void exec() {
        try {
            DataSource trainSource = new DataSource("data/Num80_90.arff");
            Instances trainData = trainSource.getDataSet();
            trainData.setClassIndex(trainData.numAttributes() - 1);

            DataSource testSource = new DataSource("data/Num20_90.arff");
            Instances testData = testSource.getDataSet();
            testData.setClassIndex(testData.numAttributes() - 1);

            SMOreg smo = new SMOreg();

            Puk pukKernel = new Puk();
            pukKernel.setOmega(1.0);
            pukKernel.setSigma(1.0);
            smo.setKernel(pukKernel);

            smo.setC(1.0);

            long trainStart = System.nanoTime();
            smo.buildClassifier(trainData);
            long trainEnd = System.nanoTime();
            double trainingTime = (trainEnd - trainStart) / 1e9;

            Evaluation eval = new Evaluation(trainData);
            long testStart = System.nanoTime();
            eval.evaluateModel(smo, testData);
            long testEnd = System.nanoTime();
            double testingTime = (testEnd - testStart) / 1e9;

            System.out.println("=== SMOreg Model ===");
            System.out.println(smo); // In thông tin mô hình

            System.out.println("Time taken to build model: " + trainingTime + " seconds");
            System.out.println("\n=== Evaluation on test set ===");
            System.out.println("Time taken to test model on supplied test set: " + testingTime + " seconds");

            System.out.println("\n=== Evaluation Results ===");
            System.out.println(eval.toSummaryString("\nSummary:\n", false));


        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    public static void main(String args[]) {
        Command cmd = new RunSMORegressionAdvanceCommand();
        cmd.exec();
    }
}