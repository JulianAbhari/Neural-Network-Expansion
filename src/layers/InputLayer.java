package layers;

public class InputLayer extends Layer {

	public InputLayer(int OUTPUT_DEPTH, int OUTPUT_WIDTH, int OUTPUT_HEIGHT) {
		super(OUTPUT_DEPTH, OUTPUT_WIDTH, OUTPUT_HEIGHT);
	}

	public void setInput(double[][][] in) {
		if (!(this.OUTPUT_DEPTH != in.length || this.OUTPUT_WIDTH != in[0].length
				|| this.OUTPUT_HEIGHT != in[0][0].length)) {
			this.outputValues = in;
		}
	}

	@Override
	protected void onBuild() throws Exception {

	}

	@Override
	protected void calculateOutputDimensions() {

	}

	@Override
	public void calculate() {

	}

	@Override
	public void backpropError() {

	}

	@Override
	public void updateWeights(double learningRate) {

	}
}
