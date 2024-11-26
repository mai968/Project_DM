package cmd;

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class RunNaiveBayesClassifierCommand implements Command {
	public void exec() {
		try {
			// Load the training and test datasets
			DataSource trainSource = new DataSource("data/String80_90.arff");
			Instances trainDataset = trainSource.getDataSet();
			trainDataset.setClassIndex(trainDataset.numAttributes() - 1);

			DataSource testSource = new DataSource("data/String20_90.arff");
			Instances testDataset = testSource.getDataSet();
			testDataset.setClassIndex(testDataset.numAttributes() - 1);

			// Create and configure Naive Bayes model
			NaiveBayes model = new NaiveBayes();
			model.setOptions(new String[] { "" });

			// Measure time taken to build the model
			long trainStart = System.nanoTime();
			model.buildClassifier(trainDataset);
			long trainEnd = System.nanoTime();
			double trainTimeSeconds = (trainEnd - trainStart) / 1e9;

			// Measure time taken to evaluate the model
			long testStart = System.nanoTime();
			Evaluation eval = new Evaluation(trainDataset);
			eval.evaluateModel(model, testDataset);
			long testEnd = System.nanoTime();
			double testTimeSeconds = (testEnd - testStart) / 1e9;

			// Print the model and its evaluation results
			System.out.println("=== Naive Bayes Model ===\n");
			System.out.println(model);

			// Print training and testing time
			System.out.printf("\nTime taken to build model: %.2f seconds\n", trainTimeSeconds);
			System.out.printf("Time taken to test model on supplied test set: %.2f seconds\n", testTimeSeconds);

			System.out.println("\n=== Evaluation Results ===");
			System.out.println(eval.toSummaryString()); // Print evaluation metrics
			System.out.println(eval.toClassDetailsString()); // Class-level details

			// Print confusion matrix
			System.out.println(eval.toMatrixString()); // Print confusion matrix

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public static void main(String args[]) {
		Command cmd = new RunNaiveBayesClassifierCommand();
		cmd.exec();
	}
}
