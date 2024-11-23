package cmd;

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class RunNaiveBayesClassifierCommand implements Command {
	public void exec() {
		try {
			// Load the training and test datasets
			DataSource trainSource = new DataSource("data/String80_90.arff"); // Replace with your training dataset path
			Instances trainDataset = trainSource.getDataSet();
			trainDataset.setClassIndex(trainDataset.numAttributes() - 1); // Set the class index for training data

			DataSource testSource = new DataSource("data/String20_90.arff"); // Replace with your testing dataset path
			Instances testDataset = testSource.getDataSet();
			testDataset.setClassIndex(testDataset.numAttributes() - 1); // Set the class index for testing data

			// Create and configure Naive Bayes model
			NaiveBayes model = new NaiveBayes();
			model.setOptions(new String[] { "-K" }); // Use the default options for Naive Bayes

			// Train the model on the training dataset
			model.buildClassifier(trainDataset);

			// Evaluate the model on the test dataset
			Evaluation eval = new Evaluation(trainDataset);
			eval.evaluateModel(model, testDataset);

			// Print the model and its confusion matrix
			System.out.println("=== Naive Bayes Model ===\n");
			System.out.println(model);
			System.out.println("\n=== Evaluation Results ===");
			System.out.println(eval.toSummaryString());  // Print evaluation metrics
			System.out.println(eval.toClassDetailsString());  // Class-level details

			// Print confusion matrix
			System.out.println("\n=== Confusion Matrix ===");
			System.out.println(eval.toMatrixString());  // Print confusion matrix

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public static void main(String args[]) {
		Command cmd = new RunNaiveBayesClassifierCommand();
		cmd.exec();
	}
}