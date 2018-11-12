package functions.errorFunction;

import layers.OutputLayer;

public class CrossEntropy extends ErrorFunction {

	public CrossEntropy() {
	}

	@Override
	public double overallError(OutputLayer outputLayer, double[][][] expected) {
		double val = 0;
		double c = 0;

		for (int i = 0; i < outputLayer.getOutputValues().length; i++) {
			for (int n = 0; n < outputLayer.getOutputValues()[0].length; n++) {
				for (int j = 0; j < outputLayer.getOutputValues()[0][0].length; j++) {
					val += expected[i][n][j] * Math.log(outputLayer.getOutputValues()[i][n][j])
							+ (1 - expected[i][n][j]) * Math.log(1 - outputLayer.getOutputValues()[i][n][j]);
					c++;
				}
			}
		}

		return -(1 / c) * val;
	}

	@Override
	public void apply(OutputLayer outputLayer, double[][][] expected) {
		for (int i = 0; i < outputLayer.getOutputValues().length; i++) {
			for (int n = 0; n < outputLayer.getOutputValues()[0].length; n++) {
				for (int j = 0; j < outputLayer.getOutputValues()[0][0].length; j++) {
					outputLayer.getOutputErrorValues()[i][n][j] = -(expected[i][n][j]
							- outputLayer.getOutputValues()[i][n][j]);
				}
			}
		}
	}

}
