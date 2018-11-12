package functions.errorFunction;

import layers.OutputLayer;

public class MeanSquaredError extends ErrorFunction {

	public double overallError(OutputLayer outputLayer, double[][][] expectedOutput) {
		double sum = 0;
		double numOfOutputNeurons = 0;

		for (int i = 0; i < outputLayer.getOutputValues().length; i += 1) {
			for (int j = 0; j < outputLayer.getOutputValues()[0].length; j += 1) {
				for (int k = 0; k < outputLayer.getOutputValues()[0][0].length; k += 1) {
					sum += (outputLayer.getOutputValues()[i][j][k] - expectedOutput[i][j][k])
							* (outputLayer.getOutputValues()[i][j][k] - expectedOutput[i][j][k]);
					numOfOutputNeurons += 1;
				}
			}
		}

		return sum / (2.0 * numOfOutputNeurons);
	}

	public void apply(OutputLayer outputLayer, double[][][] expectedOutput) {
		double[][][] output = outputLayer.getOutputValues();
		double[][][] outputDerivative = outputLayer.getOutputDerivativeValues();
		double[][][] errorSignals = outputLayer.getOutputErrorValues();
		
		for (int i = 0; i < outputLayer.getOutputValues().length; i += 1) {
			for (int j = 0; j < outputLayer.getOutputValues()[0].length; j += 1) {
				for (int k = 0; k < outputLayer.getOutputValues()[0][0].length; k += 1) {
					errorSignals[i][j][k] = outputDerivative[i][j][k] * (output[i][j][k] - expectedOutput[i][j][k]);
				}
			}
		}
		
	}

}
