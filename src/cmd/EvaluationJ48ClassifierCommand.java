//package cmd;
//
//import util.Loader;
////import util.Printer;
//import weka.classifiers.evaluation.Evaluation;
//import weka.classifiers.trees.J48;
//import weka.core.Instances;
//
//public class EvaluationJ48ClassifierCommand implements Command {
//    public void exec() {
//        // Load the dataset
//        Instances dataset = Loader.loadArff("data/String10.arff"); // Replace with your dataset path
//
//        // Set the class index
//        dataset.setClassIndex(dataset.numAttributes() - 1);
//
//        try {
//            // Create and configure J48 decision tree
//            J48 tree = new J48();
//            tree.setOptions(new String[] { "-C", "0.25", "-M", "10" }); // Configure the tree
//
//            // Train the model on the entire dataset
//            tree.buildClassifier(dataset);
//
//            // Evaluate the model using cross-validation
//            Evaluation eval = new Evaluation(dataset);
//            eval.crossValidateModel(tree, dataset, 10, new java.util.Random(1)); // 10-fold cross-validation
//
//            // Print the decision tree and evaluation results
//            System.out.println("=== J48 Model ===\n");
//            System.out.println(tree);
//            System.out.println(tree.graph());  // Print the graph of the decision tree
//            System.out.println("\n=== Evaluation Results ===");
//            System.out.println(eval.toSummaryString());  // Print evaluation summary
//            System.out.println(eval.toClassDetailsString());  // Print class-level evaluation details
//
//            // Print confusion matrix
//            System.out.println("\n=== Confusion Matrix ===");
//            System.out.println(eval.toMatrixString());  // Print confusion matrix
//
//
//        } catch (Exception e) {
//            e.printStackTrace();
//        }
//    }
//
//    public static void main(String args[]) {
//        Command cmd = new EvaluationJ48ClassifierCommand();
//        cmd.exec();
//    }
//}
//
//
//
