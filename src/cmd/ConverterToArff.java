package cmd;

import util.Saver;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import java.io.File;

public class ConverterToArff implements Command {
    public final static String INPUT_CSV_DATASET = "data/String10.csv"; // File CSV đầu vào
    public final static String OUTPUT_ARFF_DATASET = "data/output_String10.arff"; // File ARFF đầu ra

    public void exec() {
        try {
            // Load dataset từ file CSV
            CSVLoader csvLoader = new CSVLoader();
            csvLoader.setSource(new File(INPUT_CSV_DATASET));
            Instances dataset = csvLoader.getDataSet();

            // Lưu dataset dưới định dạng CSV và ARFF
            Saver.saveArff(OUTPUT_ARFF_DATASET, dataset);

            System.out.println("Data was copied successfully!");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String args[]) {
        Command cmd = new ConverterToArff();
        cmd.exec();
    }
}
