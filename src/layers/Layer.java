package layers;

import java.math.RoundingMode;
import java.text.DecimalFormat;

public abstract class Layer {

    protected int INPUT_DEPTH, INPUT_WIDTH, INPUT_HEIGHT;
    protected int OUTPUT_DEPTH, OUTPUT_WIDTH, OUTPUT_HEIGHT;

    private Layer prevLayer;
    private Layer nextLayer;

    protected double[][][] outputValues;
    protected double[][][] outputDerivativeValues;
    protected double[][][] outputErrorValues;

    public void connectToPreviousLayer(Layer prevLayer) throws Exception {
        this.prevLayer = prevLayer;
        this.prevLayer.nextLayer = this;
        this.INPUT_DEPTH = prevLayer.OUTPUT_DEPTH;
        this.INPUT_WIDTH = prevLayer.OUTPUT_WIDTH;
        this.INPUT_HEIGHT = prevLayer.OUTPUT_HEIGHT;

        if (this.OUTPUT_DEPTH == 0 || this.OUTPUT_WIDTH == 0 || this.OUTPUT_HEIGHT == 0) {
            calculateOutputDimensions();
        }

        if (this.OUTPUT_DEPTH < 1 || this.OUTPUT_HEIGHT < 1 || this.OUTPUT_WIDTH < 1) {
            throw new Exception("Bad Dimensions!");
        }

        initializeArrays();
        onBuild();
    }

    private void initializeArrays() {
        outputValues = new double[OUTPUT_DEPTH][OUTPUT_WIDTH][OUTPUT_HEIGHT];
        outputDerivativeValues = new double[OUTPUT_DEPTH][OUTPUT_WIDTH][OUTPUT_HEIGHT];
        outputErrorValues = new double[OUTPUT_DEPTH][OUTPUT_WIDTH][OUTPUT_HEIGHT];
    }

    protected abstract void onBuild() throws Exception;

    protected abstract void calculateOutputDimensions() throws Exception;

    public Layer(int OUTPUT_DEPTH, int OUTPUT_WIDTH, int OUTPUT_HEIGHT) {
        this.OUTPUT_DEPTH = OUTPUT_DEPTH;
        this.OUTPUT_WIDTH = OUTPUT_WIDTH;
        this.OUTPUT_HEIGHT = OUTPUT_HEIGHT;
    }

    public Layer() {

    }

    public abstract void calculate();

    public abstract void backpropError();

    public abstract void updateWeights(double learningRate);

    public int getINPUT_DEPTH() {
        return INPUT_DEPTH;
    }

    public int getINPUT_WIDTH() {
        return INPUT_WIDTH;
    }

    public int getINPUT_HEIGHT() {
        return INPUT_HEIGHT;
    }

    public int getOUTPUT_DEPTH() {
        return OUTPUT_DEPTH;
    }

    public int getOUTPUT_WIDTH() {
        return OUTPUT_WIDTH;
    }

    public int getOUTPUT_HEIGHT() {
        return OUTPUT_HEIGHT;
    }

    public Layer getPrevLayer() {
        return prevLayer;
    }

    public Layer getNextLayer() {
        return nextLayer;
    }

    public double[][][] getOutputValues() {
        return outputValues;
    }

    public double[][][] getOutputDerivativeValues() {
        return outputDerivativeValues;
    }

    public double[][][] getOutputErrorValues() {
        return outputErrorValues;
    }

    public double[][][] getInputValues() {
        return prevLayer.outputValues;
    }

    public double[][][] getInputDerivativeValues() {
        return prevLayer.outputDerivativeValues;
    }

    public double[][][] getInputErrorValues() {
        return prevLayer.outputErrorValues;
    }

    public void setOutputValues(double[][][] outputValues) {
        if (matchingDimensions(outputValues))
            this.outputValues = outputValues;
    }

    public void setOutputDerivativeValues(double[][][] outputDerivativeValues) {
        if (matchingDimensions(outputDerivativeValues))
            this.outputDerivativeValues = outputDerivativeValues;
    }

    public void setOutputErrorValues(double[][][] outputErrorValues) {
        if (matchingDimensions(outputErrorValues))
            this.outputErrorValues = outputErrorValues;
    }

    public static boolean matchingDimensions(Layer layer, double[][][] values) {
        return layer.getOUTPUT_DEPTH() == values.length &&
                layer.getOUTPUT_WIDTH() == values[0].length &&
                layer.getOUTPUT_HEIGHT() == values[0][0].length;
    }

    public boolean matchingDimensions(double[][][] values) {
        return matchingDimensions(this, values);
    }

    public static void printArray(double[][][] array) {
        DecimalFormat df = new DecimalFormat("#.###");
        df.setRoundingMode(RoundingMode.CEILING);
        if (array.length == 1 && array[0].length == 1) {
            String s = "";
            for (int i = 0; i < array[0][0].length; i++) {
                s+= String.format("%-8s",df.format(array[0][0][i]))+ " ";
            }
            System.out.println(s);
        } else {
            for (int n = 0; n < array[0][0].length; n++) {
                String s = "";
                for (int i = 0; i < array.length; i++) {
                    for (int k = 0; k < array[0].length; k++) {
                        s+= String.format("%-8s",df.format(array[i][k][n]))+ " ";
                    }
                    s += "       ";
                }
                System.out.println(s);
            }
        }
    }

    public void printOutput() {
        printArray(this.outputValues);
    }

    public void printErrorValues() {
        printArray(this.getOutputErrorValues());
    }

    public void printOutputDerivative() {
        printArray(this.getOutputDerivativeValues());
    }

    public void feedForwardRecursive() {
        this.calculate();
        if (this.nextLayer != null) {
            this.nextLayer.feedForwardRecursive();
        }
    }

    public void backpropagateErrorRecursive() {
        this.backpropError();
        if (this.prevLayer != null) {
            this.prevLayer.backpropagateErrorRecursive();
        }
    }

    public void updateWeightsRecursive(double learningRate) {
        this.updateWeights(learningRate);
        if (this.nextLayer != null) {
            this.nextLayer.updateWeightsRecursive(learningRate);
        }
    }

    @Override
    public String toString() {
        return this.getClass().getSimpleName() + " {" +
                "  INPUT= " + INPUT_DEPTH +
                ", " + INPUT_WIDTH +
                ", " + INPUT_HEIGHT +
                "  OUTPUT= " + OUTPUT_DEPTH +
                ", " + OUTPUT_WIDTH +
                ", " + OUTPUT_HEIGHT +
                '}';
    }
}
