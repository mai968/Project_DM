package cmd;

import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.rules.ZeroR; // Import ZeroR
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class RunZeroRClassifierCommand implements Command {
    public void exec() {
        try {
            DataSource trainSource = new DataSource("data/String80_90.arff");
            Instances trainDataset = trainSource.getDataSet();
            trainDataset.setClassIndex(trainDataset.numAttributes() - 1);

            DataSource testSource = new DataSource("data/String20_90.arff");
            Instances testDataset = testSource.getDataSet();
            testDataset.setClassIndex(testDataset.numAttributes() - 1);

            ZeroR zeroR = new ZeroR();

            zeroR.buildClassifier(trainDataset);

            Evaluation eval = new Evaluation(trainDataset);
            eval.evaluateModel(zeroR, testDataset);

            System.out.println("=== ZeroR Model ===\n");
            System.out.println(zeroR);
            System.out.println("\n=== Evaluation Results ===");
            System.out.println(eval.toSummaryString());
            System.out.println(eval.toClassDetailsString());

            System.out.println("\n=== Confusion Matrix ===");
            System.out.println(eval.toMatrixString());

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String args[]) {
        Command cmd = new RunZeroRClassifierCommand();
        cmd.exec();
    }
}
