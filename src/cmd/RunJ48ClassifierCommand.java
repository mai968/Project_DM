package cmd;

import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class RunJ48ClassifierCommand implements Command {
	public void exec() {
		try {
			// Load the training and test datasets
			DataSource trainSource = new DataSource("data/String80_90.arff");
			Instances trainDataset = trainSource.getDataSet();
			trainDataset.setClassIndex(trainDataset.numAttributes() - 1);

			DataSource testSource = new DataSource("data/String20_90.arff");
			Instances testDataset = testSource.getDataSet();
			testDataset.setClassIndex(testDataset.numAttributes() - 1);

 			// Create and configure J48 model
			J48 tree = new J48();
			tree.setOptions(new String[]{"-C", "0.25", "-M", "10"});

			// Training the model and measure time to build the model
			long trainStart = System.nanoTime();
			tree.buildClassifier(trainDataset);
			long trainEnd = System.nanoTime();
			double trainingTime = (trainEnd - trainStart) / 1e9;

			// Evaluate the model and measure the evaluation time.
			Evaluation eval = new Evaluation(trainDataset);
			long testStart = System.nanoTime();
			eval.evaluateModel(tree, testDataset);
			long testEnd = System.nanoTime();
			double testingTime = (testEnd - testStart) / 1e9;

			// Print the model and its evaluation results
			System.out.println("=== J48 Model ===\n");
			System.out.println(tree);
			System.out.println(tree.graph());

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
		Command cmd = new RunJ48ClassifierCommand();
		cmd.exec();
	}
}
