package strategy;

import java.util.ArrayList;
import java.util.Random;

import agent.AgentAction;
import motor.PacmanGame;
import neuralNetwork.TrainExample;

public class ApproximateQLearningStrategy extends QLearningStrategy {

	private double[] weights; // Weights for the feature functions
	private int numberOfFeatures; // Number of features used for the approximation

	public ApproximateQLearningStrategy(double epsilon, double gamma, double alpha, int sizeMazeX, int sizeMazeY) {
		super(epsilon, gamma, alpha, sizeMazeX, sizeMazeY);
		this.numberOfFeatures = 1; // For simplicity, we're considering only 1 feature here
		this.weights = new double[numberOfFeatures];
		for (int i = 0; i < numberOfFeatures; i++) {
			weights[i] = Math.random() * 0.01; // Small random initialization
		}
	}

	// Compute the approximate Q value for a given state and action
	private double getQValue(PacmanGame state, AgentAction action) {
		double qValue = 0.0;
		double[] features = extractFeatures(state, action);
		for (int i = 0; i < numberOfFeatures; i++) {
			qValue += weights[i] * features[i];
		}
		return qValue;
	}

	// Extract features from the state and action (to be implemented)
	private double[] extractFeatures(PacmanGame state, AgentAction action) {
		double[] features = new double[numberOfFeatures];

		// Example feature: Simplistic representation for demonstration
		// Actual implementation would involve more complex logic to calculate features
		features[0] = calculateDistanceToNearestDot(state, action); // Placeholder method

		return features;
	}

	// Placeholder method for calculating the distance to the nearest dot
	private double calculateDistanceToNearestDot(PacmanGame state, AgentAction action) {
		// Implement logic to calculate distance to the nearest dot
		// This is highly simplified and should be replaced with actual logic
		return Math.random() * 10; // Random placeholder
	}

	@Override
	public AgentAction chooseAction(PacmanGame state) {
		Random rand = new Random();
		if (rand.nextDouble() < current_epsilon) {
			// Exploration: choose a random action from legal actions
			ArrayList<AgentAction> legalActions = state.getLegalPacmanActions();
			return legalActions.get(rand.nextInt(legalActions.size()));
		} else {
			// Exploitation: choose the best action based on Q values
			double bestValue = Double.NEGATIVE_INFINITY;
			AgentAction bestAction = null;
			for (AgentAction action : state.getLegalPacmanActions()) {
				double value = getQValue(state, action);
				if (value > bestValue) {
					bestValue = value;
					bestAction = action;
				}
			}
			return bestAction != null ? bestAction : state.getLegalPacmanActions().get(0);
		}
	}

	@Override
	public void update(PacmanGame state, PacmanGame nextState, AgentAction action, double reward, boolean isFinalState) {
		double[] features = extractFeatures(state, action);
		double qValue = getQValue(state, action);
		double maxQValueNext = Double.NEGATIVE_INFINITY;
		for (AgentAction nextAction : nextState.getLegalPacmanActions()) {
			double value = getQValue(nextState, nextAction);
			if (value > maxQValueNext) {
				maxQValueNext = value;
			}
		}
		double difference = (reward + gamma * (isFinalState ? 0 : maxQValueNext)) - qValue;

		// Update weights for each feature based on the difference
		for (int i = 0; i < numberOfFeatures; i++) {
			weights[i] += learningRate * difference * features[i];
		}
	}

	@Override
	public void learn(ArrayList<TrainExample> trainExamples) {
		// Not used in this strategy
	}
}
