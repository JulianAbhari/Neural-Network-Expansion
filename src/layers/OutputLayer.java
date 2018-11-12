package layers;

import functions.errorFunction.ErrorFunction;
import functions.errorFunction.MeanSquaredError;

public class OutputLayer extends Layer {

    public OutputLayer(int OUTPUT_DEPTH, int OUTPUT_WIDTH, int OUTPUT_HEIGHT) {
        super(OUTPUT_DEPTH, OUTPUT_WIDTH, OUTPUT_HEIGHT);
    }
    public OutputLayer(Layer prev_layer) {
        this(prev_layer.OUTPUT_DEPTH, prev_layer.OUTPUT_WIDTH, prev_layer.OUTPUT_HEIGHT);
    }

    private ErrorFunction errorFunction;

    public ErrorFunction getErrorFunction() {
        return errorFunction;
    }

    public void setErrorFunction(ErrorFunction errorFunction) {
        this.errorFunction = errorFunction;
    }

    public void calculateOutputErrorValues(double[][][] expectedOutput) {
        this.errorFunction.apply(this, expectedOutput);
    }

    public double overallError(double[][][] expected) {
        return this.getErrorFunction().overallError(this,expected);
    }

    @Override
    protected void onBuild() {
        if(this.errorFunction == null) this.errorFunction = new MeanSquaredError();
    }

    @Override
    protected void calculateOutputDimensions() {

    }

    @Override
    public void calculate() {
        this.outputValues = this.getInputValues();
        this.outputDerivativeValues = this.getInputDerivativeValues();
    }

    @Override
    public void backpropError() {
        this.getPrevLayer().setOutputErrorValues(this.outputErrorValues);
    }

    @Override
    public void updateWeights(double learningRate) {

    }
}

