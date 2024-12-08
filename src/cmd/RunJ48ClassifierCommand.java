package cmd;

import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class RunJ48ClassifierCommand implements Command {
	public void exec() {
		try {
			DataSource trainSource = new DataSource("data/String80_90.arff");
			Instances trainDataset = trainSource.getDataSet();
			trainDataset.setClassIndex(trainDataset.numAttributes() - 1);

			DataSource testSource = new DataSource("data/String20_90.arff");
			Instances testDataset = testSource.getDataSet();
			testDataset.setClassIndex(testDataset.numAttributes() - 1);

			J48 tree = new J48();
			tree.setOptions(new String[]{"-C", "0.25", "-M", "10"});

			tree.buildClassifier(trainDataset);

			Evaluation eval = new Evaluation(trainDataset);
			eval.evaluateModel(tree, testDataset);

			System.out.println("=== J48 Model ===\n");
			System.out.println(tree);
			System.out.println(tree.graph());
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
		Command cmd = new RunJ48ClassifierCommand();
		cmd.exec();
	}
}
