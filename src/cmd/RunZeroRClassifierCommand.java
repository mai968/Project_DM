package cmd;

import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.rules.ZeroR;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class RunZeroRClassifierCommand implements Command {
    public void exec() {
        try {
            // Load the training and test datasets
            DataSource trainSource = new DataSource("data/String80_90.arff"); // Thay bằng đường dẫn tập train
            Instances trainDataset = trainSource.getDataSet();
            trainDataset.setClassIndex(trainDataset.numAttributes() - 1); // Đặt chỉ số lớp cho tập train

            DataSource testSource = new DataSource("data/String20_90.arff"); // Thay bằng đường dẫn tập test
            Instances testDataset = testSource.getDataSet();
            testDataset.setClassIndex(testDataset.numAttributes() - 1); // Đặt chỉ số lớp cho tập test

            // Create and configure ZeroR model
            ZeroR zeroR = new ZeroR();

            // Training the model and measure time to build the model
            long trainStart = System.nanoTime();
            zeroR.buildClassifier(trainDataset);
            long trainEnd = System.nanoTime();
            double trainingTime = (trainEnd - trainStart) / 1e9;

            // Evaluate the model and measure the evaluation time
            Evaluation eval = new Evaluation(trainDataset);
            long testStart = System.nanoTime();
            eval.evaluateModel(zeroR, testDataset);
            long testEnd = System.nanoTime();
            double testingTime = (testEnd - testStart) / 1e9;

            // Print the model and its evaluation results
            System.out.println("=== ZeroR Model ===\n");
            System.out.println(zeroR);

            System.out.printf("\nTime taken to build model: %.2f seconds\n", trainingTime);
            System.out.println("\n=== Evaluation on test set ===");
            System.out.printf("Time taken to test model on supplied test set: %.2f seconds\n", testingTime);

            System.out.println("\n=== Evaluation Results ===");
            System.out.println(eval.toSummaryString());
            System.out.println(eval.toClassDetailsString());
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
