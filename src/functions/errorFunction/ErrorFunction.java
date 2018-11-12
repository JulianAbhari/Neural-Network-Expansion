package functions.errorFunction;

import layers.OutputLayer;

public abstract class ErrorFunction {
	
	public abstract double overallError(OutputLayer outputLayer, double[][][] expectedOutput);
	
	public abstract void apply(OutputLayer outputLayer, double[][][] expectedOutput);
}
