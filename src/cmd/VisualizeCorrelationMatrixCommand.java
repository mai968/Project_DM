package cmd;

import util.Loader;
import weka.attributeSelection.CorrelationAttributeEval;
import weka.core.Instances;

public class VisualizeCorrelationMatrixCommand implements Command {
    public void exec() {
        // Load the dataset from the CSV file
        Instances dataset = Loader.loadCsv("data/cleaned_data_Num.csv"); // Đường dẫn đến tập CSV

        try {
            CorrelationAttributeEval cEval = new CorrelationAttributeEval();
            System.out.println();

            // In tiêu đề các thuộc tính
            for (int i = dataset.numAttributes() - 1; i >= 0; i--) {
                System.out.print("A[" + i + "]" + " ");
            }

            System.out.println();

            // Tính ma trận tương quan
            for (int i = dataset.numAttributes() - 1; i >= 0; i--) {
                dataset.setClassIndex(i); // Đặt thuộc tính i làm class
                cEval.buildEvaluator(dataset); // Xây dựng bộ đánh giá tương quan

                for (int j = 0; j <= i; j++) {
                    if (j == 0) {
                        System.out.print(
                                "A[" + i + "]" + " " + String.format("%.2f", cEval.evaluateAttribute(j)) + " ");
                    } else {
                        System.out.print(String.format("%.2f", cEval.evaluateAttribute(j)) + " ");
                    }
                }
                System.out.println("\t");
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String args[]) {
        Command cmd = new VisualizeCorrelationMatrixCommand();
        cmd.exec();
    }
}
