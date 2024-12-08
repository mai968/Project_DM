import cmd.*;

public class Main {
	public static void preprocessing() {
		(new ReadCSVFileCommand()).exec();
		(new VisualizeCorrelationMatrixCommand()).exec();
	}

	public static void zeroR() { (new RunZeroRClassifierCommand()).exec();}

	public static void OneR() { (new RunOneRClassifierCommand()).exec();}

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
		zeroR();
		OneR();
		j48();
		naiveBayes();
		linearRegression();
		SMORegression();
		SMORegressionAdvance();
	}
}
