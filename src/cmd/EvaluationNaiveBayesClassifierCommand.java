//package cmd;
//
//import util.Loader;
//import weka.classifiers.evaluation.Evaluation;
//import weka.classifiers.bayes.NaiveBayes;
//import weka.core.Instances;
//
//public class EvaluationNaiveBayesClassifierCommand implements Command {
//    public void exec() {
//        // Load the dataset
//        Instances dataset = Loader.loadArff("data/String10.arff"); // Replace with your dataset path
//
//        // Set the class index
//        dataset.setClassIndex(dataset.numAttributes() - 1);
//
//        try {
//            // Create and configure NaiveBayes classifier
//            NaiveBayes naiveBayes = new NaiveBayes();
//
//
//            // Create and configure Naive Bayes model
////            NaiveBayes model = new NaiveBayes();
//            naiveBayes.setOptions(new String[] {}); // Use the default options for Naive Bayes
//
//            // Train the model on the entire dataset
//            naiveBayes.buildClassifier(dataset);
//
//
//            // Evaluate the model using cross-validation
//            Evaluation eval = new Evaluation(dataset);
//            eval.crossValidateModel(naiveBayes, dataset, 10, new java.util.Random(1)); // 10-fold cross-validation
//
//            // Print the Naive Bayes model and evaluation results
//            System.out.println("=== Naive Bayes Model ===\n");
//            System.out.println(naiveBayes);
//            System.out.println("\n=== Evaluation Results ===");
//            System.out.println(eval.toSummaryString());  // Print evaluation summary
//            System.out.println(eval.toClassDetailsString());  // Print class-level evaluation details
//
//            // Print confusion matrix
//            System.out.println("\n=== Confusion Matrix ===");
//            System.out.println(eval.toMatrixString());  // Print confusion matrix
//
//        } catch (Exception e) {
//            e.printStackTrace();
//        }
//    }
//
//    public static void main(String args[]) {
//        Command cmd = new EvaluationNaiveBayesClassifierCommand();
//        cmd.exec();
//    }
//}
