package layers;

public class PoolingLayer extends Layer {

	private int pooling_factor;

	public PoolingLayer(int pooling_factor) {
		super();
		this.pooling_factor = pooling_factor;
	}

	@Override
	protected void onBuild() throws Exception {

	}

	@Override
	protected void calculateOutputDimensions() throws Exception {
		this.OUTPUT_DEPTH = this.INPUT_DEPTH;
		this.OUTPUT_WIDTH = this.INPUT_WIDTH / pooling_factor + (this.INPUT_WIDTH % pooling_factor > 0 ? 1 : 0);
		this.OUTPUT_HEIGHT = this.INPUT_HEIGHT / pooling_factor + (this.INPUT_HEIGHT % pooling_factor > 0 ? 1 : 0);

	}

	@Override
	public void calculate() {
		for (int i = 0; i < this.OUTPUT_DEPTH; i++) {
			for (int n = 0; n < this.OUTPUT_WIDTH; n++) {
				for (int k = 0; k < this.OUTPUT_HEIGHT; k++) {
					this.outputValues[i][n][k] = 0;
					this.outputDerivativeValues[i][n][k] = 0;
				}
			}
		}
		for (int i = 0; i < this.OUTPUT_DEPTH; i++) {
			for (int n = 0; n < this.OUTPUT_WIDTH; n++) {
				for (int k = 0; k < this.OUTPUT_HEIGHT; k++) {

					double max = 0;
					double d = 0;

					for (int x = 0; x < pooling_factor; x++) {
						for (int y = 0; y < pooling_factor; y++) {

							int x_i = n * pooling_factor + x;
							int y_i = k * pooling_factor + y;

							if (x_i < this.INPUT_WIDTH && y_i < this.INPUT_HEIGHT) {
								if (getInputValues()[i][x_i][y_i] > max) {
									max = getInputValues()[i][x_i][y_i];
									d = getInputDerivativeValues()[i][x_i][y_i];
								}
							}
						}
					}

					outputValues[i][n][k] = max;
					outputDerivativeValues[i][n][k] = d;

				}
			}
		}
	}

	@Override
	public void backpropError() {
		for (int i = 0; i < this.OUTPUT_DEPTH; i++) {
			for (int n = 0; n < this.OUTPUT_WIDTH; n++) {
				for (int k = 0; k < this.OUTPUT_HEIGHT; k++) {

					for (int x = 0; x < pooling_factor; x++) {
						for (int y = 0; y < pooling_factor; y++) {

							int x_i = n * pooling_factor + x;
							int y_i = k * pooling_factor + y;

							if (x_i < this.INPUT_WIDTH && y_i < this.INPUT_HEIGHT) {
								if (getInputValues()[i][x_i][y_i] == outputValues[i][n][k]) {
									this.getPrevLayer().outputErrorValues[i][x_i][y_i] = outputErrorValues[i][n][k];
								}
							}
						}
					}
				}
			}
		}
	}

	@Override
	public void updateWeights(double learningRate) {
	}

	public int getPooling_factor() {
		return pooling_factor;
	}
}