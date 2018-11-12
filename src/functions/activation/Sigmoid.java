package functions.activation;

public class Sigmoid extends ActivationFunction {

	public double activation(double number) {
		return 1.0 / (1 + Math.exp(-number));
	}

	public double activationDerivative(double number) {
		return activation(number) * (1 - activation(number));
	}

}
