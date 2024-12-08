package cmd;


import weka.classifiers.rules.OneR;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;


public class RunOneRClassifierCommand implements Command {
    public void exec() {
        try {
            DataSource trainSource = new DataSource("data/String80_90.arff");
            Instances trainDataset = trainSource.getDataSet();
            trainDataset.setClassIndex(trainDataset.numAttributes() - 1);


            DataSource testSource = new DataSource("data/String20_90.arff");
            Instances testDataset = testSource.getDataSet();
            testDataset.setClassIndex(testDataset.numAttributes() - 1);


            OneR model = new OneR();
            model.setOptions(new String[] { "-B", "6" });


            long startTime = System.currentTimeMillis();
            model.buildClassifier(trainDataset);
            long endTime = System.currentTimeMillis();
            long trainingTime = endTime - startTime;


            Evaluation eval = new Evaluation(trainDataset);
            eval.evaluateModel(model, testDataset);


            System.out.println("=== OneR Model ===\n");
            System.out.println(model);
            System.out.println("\n=== Evaluation Results ===");
            System.out.println(eval.toSummaryString());
            System.out.println(eval.toClassDetailsString());


            System.out.println("\n=== Confusion Matrix ===");
            System.out.println(eval.toMatrixString());


            System.out.println("\n=== Training Time ===");
            System.out.println("Time taken to train the model: " + trainingTime + " ms");


        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    public static void main(String args[]) {
        Command cmd = new RunOneRClassifierCommand();
        cmd.exec();
    }
}

