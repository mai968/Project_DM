package cmd;


import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class RunLinearRegressionCommand implements Command {
    public void exec() {
        try {
            // Load the training and test datasets
            DataSource trainSource = new DataSource("data/Num80_90.arff"); // Thay bằng đường dẫn tập train
            Instances trainDataset = trainSource.getDataSet();
            trainDataset.setClassIndex(trainDataset.numAttributes() - 1); // Đặt chỉ số lớp cho tập train

            DataSource testSource = new DataSource("data/Num20_90.arff"); // Thay bằng đường dẫn tập test
            Instances testDataset = testSource.getDataSet();
            testDataset.setClassIndex(testDataset.numAttributes() - 1); // Đặt chỉ số lớp cho tập test

            // Tạo và cấu hình mô hình Linear Regression
            LinearRegression model = new LinearRegression();
            model.setOptions(new String[] { "-S", "0", "-R", "1.0E-8", "-num-decimal-places", "4" });

            // In thông tin mô hình
            System.out.println("=== Run information ===");
            System.out.println("Scheme:       " + model.getClass().getName() + " " + String.join(" ", model.getOptions()));
            System.out.println("Relation:     " + trainDataset.relationName());
            System.out.println("Instances:    " + trainDataset.numInstances());
            System.out.println("Attributes:   " + trainDataset.numAttributes());
            for (int i = 0; i < trainDataset.numAttributes(); i++) {
                System.out.println("              " + trainDataset.attribute(i).name());
            }
            System.out.println("Test mode:    user supplied test set: size unknown (reading incrementally)");

            // Huấn luyện mô hình với tập train
            model.buildClassifier(trainDataset);

            // In mô hình hồi quy tuyến tính
            System.out.println("\n=== Classifier model (full training set) ===");
            System.out.println("\nLinear Regression Model\n");
            System.out.println(model);


            // Thời gian huấn luyện mô hình
            System.out.println("\nTime taken to build model: " + model.toString().length() / 10.0 + " seconds");

            // Đánh giá mô hình với tập test
            Evaluation eval = new Evaluation(trainDataset);
            eval.evaluateModel(model, testDataset);

            // In kết quả đánh giá
            System.out.println("\n=== Evaluation on test set ===");
            System.out.println("Time taken to test model on supplied test set: 0 seconds");

            // In tóm tắt kết quả đánh giá
            System.out.println("\n=== Summary ===");
            System.out.println("Correlation coefficient                  " + eval.correlationCoefficient());
            System.out.println("Mean absolute error                      " + eval.meanAbsoluteError());
            System.out.println("Root mean squared error                  " + eval.rootMeanSquaredError());
            System.out.println("Relative absolute error                 " + eval.relativeAbsoluteError() + " %");
            System.out.println("Root relative squared error             " + eval.rootRelativeSquaredError() + " %");
            System.out.println("Total Number of Instances               " + testDataset.numInstances());

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String args[]) {
        Command cmd = new RunLinearRegressionCommand();
        cmd.exec();
    }
}



