package layers;

import functions.activation.ActivationFunction;
import functions.activation.Sigmoid;
import tools.ArrayTools;

public class DenseLayer extends Layer {

    private double[][] weights;
    private double[] bias;

    private ActivationFunction activationFunction;


    public DenseLayer(int OUTPUT_HEIGHT) {
        super(1, 1, OUTPUT_HEIGHT);
    }

    public DenseLayer setActivationFunction(ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
        return this;
    }

    private double lowerWeightsRange = Double.NaN, upperWeigthsRange = Double.NaN;
    private double lowerBiasRange = 0, upperBiasRange = 1;


    public DenseLayer weightsRange(double lower, double upper) {
        this.lowerWeightsRange = lower;
        this.upperWeigthsRange = upper;
        return this;
    }

    public DenseLayer biasRange(double lower, double upper) {
        this.lowerBiasRange = lower;
        this.upperBiasRange = upper;
        return this;
    }

    @Override
    protected void onBuild() throws Exception{
        if(this.INPUT_WIDTH > 1 || this.INPUT_DEPTH > 1){
            throw new Exception("Input must be flattened");
        }else{
            if(weights == null || weights.length != this.OUTPUT_HEIGHT || weights[0].length != this.INPUT_HEIGHT) {
                if (Double.isNaN(lowerWeightsRange) || Double.isNaN(upperWeigthsRange)) {
                    weights = ArrayTools.createRandomArray(this.OUTPUT_HEIGHT, this.INPUT_HEIGHT, -1d / Math.sqrt(this.INPUT_HEIGHT), 1d / Math.sqrt(this.INPUT_HEIGHT));
                } else {
                    weights = ArrayTools.createRandomArray(this.OUTPUT_HEIGHT, this.INPUT_HEIGHT, lowerWeightsRange, upperWeigthsRange);
                }
            }
            if(bias == null || bias.length != this.OUTPUT_HEIGHT){
                bias =  ArrayTools.createRandomArray(this.OUTPUT_HEIGHT, lowerBiasRange, upperBiasRange);
            }
            if(activationFunction == null) {
                activationFunction = new Sigmoid();
            }
        }
    }

    public void printWeights() {
        Layer.printArray(new double[][][]{this.weights});
        Layer.printArray(new double[][][]{{this.bias}});
    }

    @Override
    protected void calculateOutputDimensions() {

    }

    @Override
    public void calculate() {
        for(int i = 0; i < this.OUTPUT_HEIGHT; i++) {
            double sum = bias[i];
            for(int n = 0; n < this.INPUT_HEIGHT; n++) {
                sum += this.getInputValues()[0][0][n] * weights[i][n];
            }
            this.outputValues[0][0][i] = sum;
            this.outputDerivativeValues[0][0][i] = sum;
        }
        this.activationFunction.apply(this.outputValues, this.outputDerivativeValues);
    }

    @Override
    public void backpropError() {
        for(int i = 0; i < this.INPUT_HEIGHT; i++) {
            double sum = 0;
            for(int n = 0; n < this.getOUTPUT_HEIGHT(); n++) {
                sum += weights[n][i] * outputErrorValues[0][0][n];
            }
            this.getPrevLayer().getOutputErrorValues()[0][0][i] = this.getInputDerivativeValues()[0][0][i] * sum;
        }
    }

    public DenseLayer setWeights(double[][] weights){
        if(weights != null && weights[0] != null){
            this.weights = weights;
            this.lowerWeightsRange = -1;
            this.upperWeigthsRange = 1;
        }
        return this;
    }

    public DenseLayer setBias(double[] bias){
        if(bias != null && bias.length == this.OUTPUT_HEIGHT){
            this.bias = bias;
            this.lowerBiasRange = -1;
            this.upperBiasRange = 1;
        }
        return this;
    }

    public double[][] getWeights() {
        return weights;
    }

    public double[] getBias() {
        return bias;
    }

    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }

    @Override
    public void updateWeights(double learningRate) {

        for(int i = 0; i < this.OUTPUT_HEIGHT; i++) {
            double delta = - learningRate * this.outputErrorValues[0][0][i];
            bias[i] += delta;

            for(int prevNeuron = 0; prevNeuron < this.INPUT_HEIGHT; prevNeuron ++) {
                weights[i][prevNeuron] += delta * getInputValues()[0][0][prevNeuron];
            }
        }
    }

}

