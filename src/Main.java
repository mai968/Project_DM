//import cmd.CleanDataCommand;
//import cmd.DiscretizeAttributesCommand;
//import cmd.HandleMissingDataCommand;
//import cmd.OriginalCsv2ArffCommand;
//import cmd.RemoveOutlierAndExtremeDataCommand;
//import cmd.RemoveUselessAttributesCommand;
import cmd.RunJ48ClassifierCommand;
import cmd.RunNaiveBayesClassifierCommand;
import cmd.RunLinearRegressionCommand;
import cmd.RunSMORegressionCommand;
import cmd.RunSMORegressionAdvanceCommand;

//import cmd.RunOneRClassifierCommand;
//import cmd.VisualizeCorrelationMatrixCommand;

public class Main {
	public static void preprocessing() {
//		(new OriginalCsv2ArffCommand()).exec();
//		(new RemoveUselessAttributesCommand()).exec();
//		(new HandleMissingDataCommand()).exec();
//		(new RemoveOutlierAndExtremeDataCommand()).exec();
//		(new CleanDataCommand()).exec();
//		(new VisualizeCorrelationMatrixCommand()).exec();
//		(new DiscretizeAttributesCommand()).exec();
	}

//	public static void zeroR() {
//		(new RunOneRClassifierCommand()).exec();
//	}

	public static void j48() {
		(new RunJ48ClassifierCommand()).exec();
	}

	public static void naiveBayes() {
		(new RunNaiveBayesClassifierCommand()).exec();
	}

	public static void linearRegression() {
		(new RunLinearRegressionCommand()).exec();
	}

	public static void SMORegression() {
		(new RunSMORegressionCommand()).exec();
	}

	public static void SMORegressionAdvance() {(new RunSMORegressionAdvanceCommand()).exec();}

	public static void main(String args[]) {
		preprocessing();
//		zeroR();
		j48();
		naiveBayes();
		linearRegression();
		SMORegression();
		SMORegressionAdvance();
	}
}
