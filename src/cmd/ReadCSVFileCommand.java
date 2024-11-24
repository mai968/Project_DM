package cmd;

import util.Saver;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import java.io.File;

public class ReadCSVFileCommand implements Command {
    public final static String csvFilePath = "data/cleaned_data_Num.csv"; // Đường dẫn file CSV

    public void exec() {
        try {
            // Sử dụng CSVLoader để đọc file CSV
            CSVLoader csvLoader = new CSVLoader();
            csvLoader.setSource(new File(csvFilePath));
            Instances dataset = csvLoader.getDataSet();

            // In dữ liệu ra console
            System.out.println("Number of row: " + dataset.numInstances());
            System.out.println("Number of column: " + dataset.numAttributes());
            System.out.println("Dataset:");
            System.out.println(dataset);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String args[]) {
        Command cmd = new ReadCSVFileCommand();
        cmd.exec();
    }
}