package network;

import java.util.ArrayList;

import data.TrainSet;
import functions.errorFunction.ErrorFunction;
import layers.DenseLayer;
import layers.InputLayer;
import layers.Layer;
import layers.OutputLayer;
import tools.ArrayTools;

/**
 * Neural Network Expansion
 * 
 * 07/17/18
 * 
 * @author Julian Abhari
 */

public class Network {

	private InputLayer inputLayer;
	private OutputLayer outputLayer;

	public Network(InputLayer inputLayer, OutputLayer outputLayer) {
		this.inputLayer = inputLayer;
		this.outputLayer = outputLayer;
	}

	public void printArchitecture() {
		Layer currentLayer = inputLayer;

		System.out.println(currentLayer.getClass().getSimpleName());
		System.out.println("Network Size: [" + currentLayer.getOUTPUT_DEPTH() + " " + currentLayer.getOUTPUT_WIDTH()
				+ " " + currentLayer.getOUTPUT_HEIGHT() + "]");

		while (currentLayer.getNextLayer() != null) {
			System.out.println("");
			currentLayer = currentLayer.getNextLayer();
			System.out.println(currentLayer.getClass().getSimpleName());
			System.out.println("Network Size: [" + currentLayer.getOUTPUT_DEPTH() + " " + currentLayer.getOUTPUT_WIDTH()
					+ " " + currentLayer.getOUTPUT_HEIGHT() + "]");
		}
	}

	public Network setErrorFunction(ErrorFunction errorFunction) {
		this.outputLayer.setErrorFunction(errorFunction);
		return this;
	}

	public double[][][] calculate(double[][][] in) {
        if (this.getInputLayer().matchingDimensions(in) == false) return null;
        this.inputLayer.setInput(in);
        this.inputLayer.feedForwardRecursive();
        return getOutput();
    }

    public void backpropagateError(double[][][] expectedOutput) {
        if (this.getOutputLayer().matchingDimensions(expectedOutput) == false) return;
        this.outputLayer.calculateOutputErrorValues(expectedOutput);
        this.outputLayer.backpropagateErrorRecursive();
    }

    public void updateWeights(double eta) {
        this.inputLayer.updateWeightsRecursive(eta);
    }

    public void train(double[][][] input, double[][][] expected, double eta) {
        if (this.getInputLayer().matchingDimensions(input) == false ||
                this.getOutputLayer().matchingDimensions(expected) == false) {
            System.err.println(
                    this.inputLayer.getOUTPUT_DEPTH() + " - " + input.length + "\n" +
                            this.inputLayer.getOUTPUT_WIDTH() + " - " + input[0].length + "\n" +
                            this.inputLayer.getOUTPUT_HEIGHT() + " - " + input[0][0].length + "\n" +
                            this.outputLayer.getOUTPUT_DEPTH() + " - " + expected.length + "\n" +
                            this.outputLayer.getOUTPUT_WIDTH() + " - " + expected[0].length + "\n" +
                            this.outputLayer.getOUTPUT_HEIGHT() + " - " + expected[0][0].length + "\n");
            return;
        }
        this.calculate(input);
        this.backpropagateError(expected);
        this.updateWeights(eta);
    }

    public void train(TrainSet trainSet, int epochs, int batch_size, double fall_of) {
        double e = 0.1;
        for (int i = 0; i < epochs; i++) {
            ArrayList<TrainSet> trainSets = trainSet.shuffledParts(batch_size);
            int index = 0;
            for (TrainSet t : trainSets) {
                index++;
                for (int k = 0; k < t.size(); k++) {
                    train(t.getInput(k), t.getOutput(k), e*fall_of);
                }
                e = overallError(t);
                System.out.println(index + "     " + e);
            }
            System.out.println("<<<<<<<<<<<<<<<<<<<<<<<<<<< " + i + " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
        }
    }


	public double overallError(TrainSet trainSet) {
        double t = 0;
        for (int i = 0; i < trainSet.size(); i++) {
            this.calculate(trainSet.getInput(i));
            t += this.getOutputLayer().overallError(trainSet.getOutput(i));
        }
        return t / (double) trainSet.size();
    }

    public double overallError(double[][][] in, double[][][] exp) {
        this.calculate(in);
        return this.getOutputLayer().overallError(exp);
    }

	public void analyseNetwork() {
		Layer currentLayer = inputLayer;
		System.out.println(currentLayer.getClass().getSimpleName());
		while (currentLayer.getNextLayer() != null) {
			System.out.println("############################################################################");
			System.out.println(currentLayer.getClass().getSimpleName());
			System.out.println("Output:");
			Layer.printArray(currentLayer.getOutputValues());
			System.out.println("Derivative:");
			Layer.printArray(currentLayer.getOutputDerivativeValues());
			System.out.println("Error:");
			Layer.printArray(currentLayer.getOutputErrorValues());
			if (currentLayer instanceof DenseLayer) {
				System.out.println("Weights:");
				Layer.printArray(new double[][][] { ((DenseLayer) currentLayer).getWeights() });
			}
			currentLayer = currentLayer.getNextLayer();
		}
	}
	
	public double[][][] getOutput() {
		return ArrayTools.copyArray(this.outputLayer.getOutputValues());
	}
	
	public InputLayer getInputLayer() {
        return inputLayer;
    }

    public OutputLayer getOutputLayer() {
        return outputLayer;
    }
    
    public int getINPUT_DEPTH() {
        return inputLayer.getOUTPUT_DEPTH();
    }

    public int getINPUT_WIDTH() {
        return inputLayer.getOUTPUT_WIDTH();
    }

    public int getINPUT_HEIGHT() {
        return inputLayer.getOUTPUT_HEIGHT();
    }

    public int getOUTPUT_DEPTH() {
        return outputLayer.getOUTPUT_DEPTH();
    }

    public int getOUTPUT_WIDTH() {
        return outputLayer.getOUTPUT_WIDTH();
    }

    public int getOUTPUT_HEIGHT() {
        return outputLayer.getOUTPUT_HEIGHT();
    }
}
