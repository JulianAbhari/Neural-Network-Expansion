package functions.activation;

public abstract class ActivationFunction {
	
	public abstract double activation(double number);
	
	public abstract double activationDerivative(double number);
	
	public void apply(double[][][] output, double[][][] outputDerivative) {
        for(int i = 0; i < output.length; i++){
            for(int n = 0; n < output[0].length; n++) {
                for(int j = 0; j < output[0][0].length; j++){
                    outputDerivative[i][n][j] = activationDerivative(output[i][n][j]);
                    output[i][n][j] = activation(output[i][n][j]);
                }
            }
        }
    }
}
