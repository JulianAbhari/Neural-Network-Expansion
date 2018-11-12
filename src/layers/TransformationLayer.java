package layers;

import network.Network;
import network.NetworkBuilder;

public class TransformationLayer extends Layer {

	@Override
	protected void onBuild() throws Exception {

	}

	@Override
	protected void calculateOutputDimensions() throws Exception {
		this.OUTPUT_DEPTH = 1;
		this.OUTPUT_WIDTH = 1;
		this.OUTPUT_HEIGHT = this.getINPUT_HEIGHT() * this.getINPUT_DEPTH() * this.getINPUT_WIDTH();
	}

	private int map(int d, int w, int h) {
		return (d * (this.getINPUT_WIDTH() * this.getINPUT_HEIGHT()) + w * this.getINPUT_HEIGHT() + h);
	}

	@Override
	public void calculate() {
		for (int i = 0; i < this.getINPUT_DEPTH(); i++) {
			for (int n = 0; n < this.getINPUT_WIDTH(); n++) {
				for (int j = 0; j < this.getINPUT_HEIGHT(); j++) {
					int index = map(i, n, j);
					this.outputValues[0][0][index] = this.getInputValues()[i][n][j];
					this.outputDerivativeValues[0][0][index] = this.getInputDerivativeValues()[i][n][j];
				}
			}
		}
	}

	@Override
	public void backpropError() {

		for (int i = 0; i < this.getINPUT_DEPTH(); i++) {
			for (int n = 0; n < this.getINPUT_WIDTH(); n++) {
				for (int j = 0; j < this.getINPUT_HEIGHT(); j++) {
					int index = map(i, n, j);
					this.getPrevLayer().outputErrorValues[i][n][j] = this.outputErrorValues[0][0][index];
				}
			}
		}
	}

	@Override
	public void updateWeights(double learningRate) {

	}

	public static void main(String[] args) {
		NetworkBuilder netBuilder = new NetworkBuilder(1, 3, 3);
		netBuilder.addLayer(new TransformationLayer());
		Network net = netBuilder.buildNetwork();

		Layer.printArray(net.calculate(new double[][][] { { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } } }));
	}
}