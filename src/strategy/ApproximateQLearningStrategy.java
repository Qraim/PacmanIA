package strategy;

import java.util.ArrayList;
import java.util.Random;

import agent.Agent;
import agent.AgentAction;
import motor.PacmanGame;
import neuralNetwork.TrainExample;

public class ApproximateQLearningStrategy extends QLearningStrategy {
	private double[] weights;
	private Random rand = new Random();

	public ApproximateQLearningStrategy(double epsilon, double gamma, double alpha, int sizeMazeX, int sizeMazeY) {
		super(epsilon, gamma, alpha, sizeMazeX, sizeMazeY);
		int numberOfFeatures = 7; // Update this if the number of features changes
		this.weights = new double[numberOfFeatures];

		// Initialize weights with random values
		for (int i = 0; i < this.weights.length; i++) {
			this.weights[i] = rand.nextDouble() * 2 - 1; // values between -1 and 1
		}
	}

	@Override
	public AgentAction chooseAction(PacmanGame state) {
		if (rand.nextDouble() < current_epsilon) {
			// Explore: choose a random action
			return new AgentAction(rand.nextInt(AgentAction.NUMBER_OF_ACTIONS));
		} else {
			// Exploit: choose the best action based on current knowledge
			double bestValue = Double.NEGATIVE_INFINITY;
			AgentAction bestAction = null;

			for (int a = 0; a < AgentAction.NUMBER_OF_ACTIONS; a++) {
				double[] features = extractFeatures(state, new AgentAction(a));
				double value = computeQValue(features);

				if (value > bestValue) {
					bestValue = value;
					bestAction = new AgentAction(a);
				}
			}
			return bestAction;
		}
	}

	@Override
	public void update(PacmanGame state, PacmanGame nextState, AgentAction action, double reward, boolean isFinalState) {
		double[] features = extractFeatures(state, action);
		double qValue = computeQValue(features);

		double maxNextQValue = isFinalState ? 0 : maxValue(nextState);
		double targetQ = reward + gamma * maxNextQValue;

		// Mise à jour des poids par descente de gradient
		for (int i = 0; i < this.weights.length; i++) {
			// Calcul de la dérivée de l'erreur par rapport à chaque poids
			double gradient = features[i] * (qValue - targetQ);
			// Mise à jour du poids
			this.weights[i] -= learningRate * gradient;
		}
	}


	private double maxValue(PacmanGame state) {
		double maxQValue = Double.NEGATIVE_INFINITY;

		for (int a = 0; a < AgentAction.NUMBER_OF_ACTIONS; a++) {
			double[] features = extractFeatures(state, new AgentAction(a));
			double qValue = computeQValue(features);

			if (qValue > maxQValue) {
				maxQValue = qValue;
			}
		}
		return maxQValue;
	}

	private double computeQValue(double[] features) {
		double qValue = 0;
		for (int i = 0; i < features.length; i++) {
			qValue += this.weights[i] * features[i];
		}
		return qValue;
	}

	private double[] extractFeatures(PacmanGame state, AgentAction action) {
		double[] features = new double[7]; // Ensure the size matches the number of features

		int nextX = state.getPacmanPosition().getX() + action.get_vx();
		int nextY = state.getPacmanPosition().getY() + action.get_vy();

		features[0] = calculateDistanceToNearestGhost(state, nextX, nextY);
		features[1] = state.getNbFood();
		features[2] = state.isCapsuleAtPosition(nextX, nextY) ? 1 : 0;
		features[3] = state.isWallAtPosition(nextX, nextY) ? 1 : 0;
		features[4] = state.isGhostsScarred() ? 1 : 0;
		features[5] = state.isLegalMovePacman(action) ? 1 : 0;
		features[6] = state.isGumAtPosition(nextX, nextY) ? 1 : 0;

		return features;
	}

	private double calculateDistanceToNearestGhost(PacmanGame state, int nextX, int nextY) {
		double nearestDistance = Double.MAX_VALUE;

		for ( Agent ghost : state.get_agentsFantom()) {
			int ghostX = ghost.get_position().getX();
			int ghostY = ghost.get_position().getY();

			double distance = Math.abs(nextX - ghostX) + Math.abs(nextY - ghostY);

			if (distance < nearestDistance) {
				nearestDistance = distance;
			}
		}
		return nearestDistance;
	}


	@Override
	public void learn(ArrayList<TrainExample> trainExamples) {
		// Implement this method if you need to train your model based on a batch of examples
	}
}
