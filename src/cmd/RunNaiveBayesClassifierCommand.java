package cmd;

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class RunNaiveBayesClassifierCommand implements Command {
	public void exec() {
		try {
			DataSource trainSource = new DataSource("data/String80_90.arff");
			Instances trainDataset = trainSource.getDataSet();
			trainDataset.setClassIndex(trainDataset.numAttributes() - 1);

			DataSource testSource = new DataSource("data/String20_90.arff");
			Instances testDataset = testSource.getDataSet();
			testDataset.setClassIndex(testDataset.numAttributes() - 1);

			NaiveBayes model = new NaiveBayes();
			model.setOptions(new String[] { "" });

			long trainStart = System.nanoTime();
			model.buildClassifier(trainDataset);
			long trainEnd = System.nanoTime();
			double trainTimeSeconds = (trainEnd - trainStart) / 1e9;

			long testStart = System.nanoTime();
			Evaluation eval = new Evaluation(trainDataset);
			eval.evaluateModel(model, testDataset);
			long testEnd = System.nanoTime();
			double testTimeSeconds = (testEnd - testStart) / 1e9;

			System.out.println("=== Naive Bayes Model ===\n");
			System.out.println(model);

			System.out.printf("\nTime taken to build model: %.2f seconds\n", trainTimeSeconds);
			System.out.printf("Time taken to test model on supplied test set: %.2f seconds\n", testTimeSeconds);

			System.out.println("\n=== Evaluation Results ===");
			System.out.println(eval.toSummaryString());
			System.out.println(eval.toClassDetailsString());
			System.out.println(eval.toMatrixString());

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public static void main(String args[]) {
		Command cmd = new RunNaiveBayesClassifierCommand();
		cmd.exec();
	}
}
