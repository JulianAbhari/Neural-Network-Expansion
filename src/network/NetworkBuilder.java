package network;

import java.util.ArrayList;

import data.TrainSet;
import functions.activation.LeakyReLU;
import functions.activation.Sigmoid;
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

public class NetworkBuilder {

	InputLayer inputLayer;

	ArrayList<Layer> layers = new ArrayList<>();

	public NetworkBuilder(int INPUT_DEPTH, int INPUT_WIDTH, int INPUT_HEIGHT) {
		inputLayer = new InputLayer(INPUT_DEPTH, INPUT_WIDTH, INPUT_HEIGHT);

		inputLayer.setOutputErrorValues(new double[INPUT_DEPTH][INPUT_WIDTH][INPUT_HEIGHT]);
		inputLayer.setOutputDerivativeValues(new double[INPUT_DEPTH][INPUT_WIDTH][INPUT_HEIGHT]);
		inputLayer.setOutputValues(new double[INPUT_DEPTH][INPUT_WIDTH][INPUT_HEIGHT]);
	}

	public NetworkBuilder addLayer(Layer layer) {
		layers.add(layer);
		return this;
	}

	public Network buildNetwork() {
		try {
			Layer prevLayer = inputLayer;
			for (Layer layer : layers) {
				layer.connectToPreviousLayer(prevLayer);
				prevLayer = layer;
			}
			OutputLayer outputLayer = new OutputLayer(prevLayer);
			outputLayer.connectToPreviousLayer(prevLayer);

			return new Network(inputLayer, outputLayer);
		} catch (Exception e) {
			e.printStackTrace();
			return null;
		}
	}

	public static void main(String[] args) {
		NetworkBuilder networkBuilder = new NetworkBuilder(1, 1, 2);
		networkBuilder.addLayer(new DenseLayer(3)
				.weightsRange(0, 1)
				.biasRange(-1, 1)
				.setActivationFunction(new Sigmoid()));
		networkBuilder.addLayer(new DenseLayer(2)
				.weightsRange(0, 1)
				.biasRange(0, 1)
				.setActivationFunction(new LeakyReLU()));
		networkBuilder.addLayer(new DenseLayer(1)
				.weightsRange(0, 1)
				.biasRange(-1, 1)
				.setActivationFunction(new Sigmoid()));
		
		Network network = networkBuilder.buildNetwork();
		
		TrainSet set = new TrainSet(1, 1, 2, 1, 1, 1);

		set.addData(ArrayTools.createComplexFlatArray(0, 0), ArrayTools.createComplexFlatArray(0));
		set.addData(ArrayTools.createComplexFlatArray(0, 1), ArrayTools.createComplexFlatArray(1));
		set.addData(ArrayTools.createComplexFlatArray(1, 0), ArrayTools.createComplexFlatArray(1));
		set.addData(ArrayTools.createComplexFlatArray(1, 1), ArrayTools.createComplexFlatArray(0));
		
		double learningRate = 0.3;
		
		for (int i = 0; i < 10; i += 1) {
			network.train(set, 10000, 4, learningRate);
		}
		
		for (int trainingExamples = 0; trainingExamples < 4; trainingExamples += 1) {
			Layer.printArray(network.calculate(set.getInput(trainingExamples)));
		}
	}
}
