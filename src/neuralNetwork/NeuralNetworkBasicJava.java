package neuralNetwork;
import java.util.ArrayList;
import java.util.List;

public class NeuralNetworkBasicJava {
	
	Matrix weights_ih,weights_ho , bias_h,bias_o;	

	
	public NeuralNetworkBasicJava(int i,int h,int o) {
		weights_ih = new Matrix(h,i);
		weights_ho = new Matrix(o,h);
		
		bias_h= new Matrix(h,1);
		bias_o= new Matrix(o,1);
		
	}
	
	public double predict(double[] X)
	{
		Matrix input = Matrix.fromArray(X);
		Matrix hidden = Matrix.multiply(weights_ih, input);
		hidden.add(bias_h);
		hidden.sigmoid();
		
		Matrix output = Matrix.multiply(weights_ho,hidden);
		output.add(bias_o);
		
		return output.toArray().get(0);
	}
	
	
	public void fit(ArrayList<TrainExample> trainExamples, int trainingStep, double learningRate)
	{
		
		double avgError = 0;
		
		for(int i=0;i<trainingStep;i++)
		{	
			if(i%1000 == 0 && i >  0) {
				System.out.println("TrainingStep : " + i + " - avgError : " + avgError/1000);
				avgError = 0;
			}
			
			int sampleN =  (int)(Math.random() * trainExamples.size() );
			
			double error  = this.train(trainExamples.get(sampleN).getX(), trainExamples.get(sampleN).getY()[0], learningRate);
			
			avgError += Math.pow(error,2);
		}
		
		
		
	}

	
	
	public double train(double [] X,double Y, double learningRate)
	{
		Matrix input = Matrix.fromArray(X);
		Matrix hidden = Matrix.multiply(weights_ih, input);
		hidden.add(bias_h);
		hidden.sigmoid();
		
		Matrix output = Matrix.multiply(weights_ho,hidden);
		output.add(bias_o);

		Matrix target = Matrix.fromDouble(Y);
		
		Matrix error = Matrix.subtract(target, output);
		
		
		double err = error.toArray().get(0);
				
		Matrix gradient = error;
		gradient.multiply(learningRate);
		
		Matrix hidden_T = Matrix.transpose(hidden);
		Matrix who_delta =  Matrix.multiply(gradient, hidden_T);
		

		
		Matrix who_T = Matrix.transpose(weights_ho);
		Matrix hidden_errors = Matrix.multiply(who_T, error);
		
		Matrix h_gradient = hidden.dsigmoid();
		h_gradient.multiply(hidden_errors);
		h_gradient.multiply(learningRate);
		
		Matrix i_T = Matrix.transpose(input);
		Matrix wih_delta = Matrix.multiply(h_gradient, i_T);
		
		weights_ho.add(who_delta);
		bias_o.add(gradient);
		
		weights_ih.add(wih_delta);
		bias_h.add(h_gradient);
		
		return err;
	}


}