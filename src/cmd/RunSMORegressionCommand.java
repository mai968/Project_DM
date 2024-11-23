package cmd;

import weka.classifiers.functions.SMOreg;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.functions.supportVector.RBFKernel; // Import kernel

public class RunSMORegressionCommand implements Command {
    public void exec() {
        try {
            // Bước 1: Load dữ liệu train và test
            DataSource trainSource = new DataSource("data/Num80_90.arff"); // Đường dẫn tập train
            Instances trainData = trainSource.getDataSet();
            trainData.setClassIndex(trainData.numAttributes() - 1); // Cột cuối là nhãn cần dự đoán

            DataSource testSource = new DataSource("data/Num20_90.arff"); // Đường dẫn tập test
            Instances testData = testSource.getDataSet();
            testData.setClassIndex(testData.numAttributes() - 1);

            // Bước 2: Tạo và cấu hình SMOreg
            SMOreg smo = new SMOreg();

            // Thiết lập kernel (ví dụ: RBF Kernel - Gaussian Kernel)
            RBFKernel rbfKernel = new RBFKernel();
            rbfKernel.setGamma(0.01); // Gamma cho RBF Kernel
            smo.setKernel(rbfKernel);

            // Cấu hình các tham số SMOreg
            smo.setC(1.0); // Regularization parameter

            // Bước 3: Đo thời gian huấn luyện mô hình
            long startTime = System.nanoTime(); // Bắt đầu đo thời gian
            smo.buildClassifier(trainData); // Huấn luyện mô hình
            long endTime = System.nanoTime(); // Kết thúc đo thời gian

            // Tính toán thời gian thực thi (giây)
            double timeTaken = (endTime - startTime) / 1e9; // Chuyển nano giây thành giây
            System.out.println("Time taken to build model: " + timeTaken + " seconds");

            // Bước 4: Đánh giá mô hình trên tập test
            Evaluation eval = new Evaluation(trainData);
            eval.evaluateModel(smo, testData);

            // Bước 5: In kết quả
            System.out.println("=== SMOreg Model ===");
            System.out.println(smo); // In thông tin mô hình

            System.out.println("Time taken to build model: " + timeTaken + " seconds");

            System.out.println("\n=== Evaluation Results ===");
            System.out.println(eval.toSummaryString("\nSummary:\n", false)); // Kết quả tóm tắt
            System.out.println("Correlation coefficient: " + eval.correlationCoefficient());
            System.out.println("Mean Absolute Error (MAE): " + eval.meanAbsoluteError());
            System.out.println("Root Mean Squared Error (RMSE): " + eval.rootMeanSquaredError());
            System.out.println("Relative Absolute Error (%): " + eval.relativeAbsoluteError());
            System.out.println("Root Relative Squared Error (%): " + eval.rootRelativeSquaredError());

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    public static void main(String args[]) {
        Command cmd = new RunSMORegressionCommand();
        cmd.exec();
    }
}
