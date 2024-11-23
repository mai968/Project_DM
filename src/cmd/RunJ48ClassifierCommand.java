package cmd;

import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class RunJ48ClassifierCommand implements Command {
	public void exec() {
		try {
			// Load the training and test datasets
			DataSource trainSource = new DataSource("data/String80_90.arff"); // Thay bằng đường dẫn tập train
			Instances trainDataset = trainSource.getDataSet();
			trainDataset.setClassIndex(trainDataset.numAttributes() - 1); // Đặt chỉ số lớp cho tập train

			DataSource testSource = new DataSource("data/String20_90.arff"); // Thay bằng đường dẫn tập test
			Instances testDataset = testSource.getDataSet();
			testDataset.setClassIndex(testDataset.numAttributes() - 1); // Đặt chỉ số lớp cho tập test

			// Tạo và cấu hình mô hình J48
			J48 tree = new J48();
			tree.setOptions(new String[]{"-C", "0.25", "-M", "10"}); // Cài đặt các tham số cho J48

			// Huấn luyện mô hình với tập train
			tree.buildClassifier(trainDataset);

			// Đánh giá mô hình với tập test
			Evaluation eval = new Evaluation(trainDataset);
			eval.evaluateModel(tree, testDataset);

			// In ra mô hình và kết quả đánh giá
			System.out.println("=== J48 Model ===\n");
			System.out.println(tree);
			System.out.println(tree.graph()); // Hiển thị cấu trúc cây quyết định
			System.out.println("\n=== Evaluation Results ===");
			System.out.println(eval.toSummaryString());  // In các số liệu đánh giá
			System.out.println(eval.toClassDetailsString());  // Chi tiết theo từng lớp

			// In ma trận nhầm lẫn
			System.out.println("\n=== Confusion Matrix ===");
			System.out.println(eval.toMatrixString());  // In ma trận nhầm lẫn

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public static void main(String args[]) {
		Command cmd = new RunJ48ClassifierCommand();
		cmd.exec();
	}
}
