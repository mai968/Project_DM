package cmd;


import weka.classifiers.rules.OneR;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;


public class RunOneRClassifierCommand implements Command {
    public void exec() {
        try {
            // Load the training and test datasets
            DataSource trainSource = new DataSource("data/String80_90.arff"); // Replace with your training dataset path
            Instances trainDataset = trainSource.getDataSet();
            trainDataset.setClassIndex(trainDataset.numAttributes() - 1); // Set the class index for training data


            DataSource testSource = new DataSource("data/String20_90.arff"); // Replace with your testing dataset path
            Instances testDataset = testSource.getDataSet();
            testDataset.setClassIndex(testDataset.numAttributes() - 1); // Set the class index for testing data


            // Create and configure OneR model
            OneR model = new OneR();
            model.setOptions(new String[] { "-B", "6" }); // Use bin size of 6 as an example (default is 6)


            // Measure the training time
            long startTime = System.currentTimeMillis();
            model.buildClassifier(trainDataset); // Train the model
            long endTime = System.currentTimeMillis();
            long trainingTime = endTime - startTime;


            // Evaluate the model on the test dataset
            Evaluation eval = new Evaluation(trainDataset);
            eval.evaluateModel(model, testDataset);


            // Print the model and its evaluation metrics
            System.out.println("=== OneR Model ===\n");
            System.out.println(model);
            System.out.println("\n=== Evaluation Results ===");
            System.out.println(eval.toSummaryString());  // Print evaluation metrics
            System.out.println(eval.toClassDetailsString());  // Class-level details


            // Print confusion matrix
            System.out.println("\n=== Confusion Matrix ===");
            System.out.println(eval.toMatrixString());  // Print confusion matrix


            // Print training time
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

