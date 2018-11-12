package functions.activation;

public class LeakyReLU extends ActivationFunction {

	public double activation(double number) {
		if (number >= 0) {
			return number;
		}
		return 0.01 * number;
	}

	public double activationDerivative(double number) {
		if (number >= 0) {
			return 1;
		}
		return 0.01;
	}

}
