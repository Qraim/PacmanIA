package strategy;

import java.util.ArrayList;
import java.util.Random;

import agent.Agent;
import agent.AgentAction;
import agent.PositionAgent;
import motor.Maze;
import motor.PacmanGame;
import neuralNetwork.TrainExample;

import java.util.HashMap;


public class TabularQLearning  extends QLearningStrategy{


	HashMap<String, double[]> QTable;



	int sizeMazeX;
	int sizeMazeY;




	public TabularQLearning( double epsilon, double gamma, double alpha,  int sizeMazeX, int sizeMazeY, int nbWalls) {
		
		super( epsilon, gamma, alpha, sizeMazeX, sizeMazeY);

		this.sizeMazeX = sizeMazeX;
		this.sizeMazeY = sizeMazeY;

		QTable = new HashMap<>();

	}




	public String encodeState(PacmanGame pacmanGame) {

		StringBuilder state = new StringBuilder();

		for (int i = 0; i < sizeMazeX; i++) {
			for (int j = 0; j < sizeMazeY; j++) {
				if (pacmanGame.isWallAtPosition(i, j)) {
					state.append("0");
				} else if (pacmanGame.isCapsuleAtPosition(i, j)) {
					state.append("1");
				} else if (pacmanGame.isPacmanAtPosition(i, j)) {
					state.append("2");
				} else if (pacmanGame.isGhostAtPosition(i, j)) {
					state.append("3");
				} else if (pacmanGame.isGumAtPosition(i, j)) {
					state.append("4");
				}  else {
					state.append("5");
				}
			}
		}
		return state.toString();

	}



	@Override
	public AgentAction chooseAction(PacmanGame state) {
		Random rand = new Random();

		if (rand.nextDouble() < current_epsilon) {
			ArrayList<AgentAction> legalmove = state.getLegalPacmanActions();
			return legalmove.get(rand.nextInt(legalmove.size()));
		} else {

			String currentState = encodeState(state);
			double[] qValues = QTable.getOrDefault(currentState, new double[]{0, 0, 0, 0});
			int bestAction = 0;
			for (int i = 1; i < qValues.length; i++) {
				if (qValues[i] > qValues[bestAction]) {
					bestAction = i;
				}
			}
			return new AgentAction(bestAction);
		}
	}



	@Override
	public void update(PacmanGame state, PacmanGame nextState, AgentAction action, double reward, boolean isFinalState) {
		String currentState = encodeState(state);
		String nextStateEncoded = encodeState(nextState);

		double[] qValuesCurrent = QTable.getOrDefault(currentState, new double[]{0, 0, 0, 0});
		double[] qValuesNext = QTable.getOrDefault(nextStateEncoded, new double[]{0, 0, 0, 0});

		double maxQNext = Double.NEGATIVE_INFINITY;

		for (double value : qValuesNext) {
			maxQNext = Math.max(maxQNext, value);
		}

		if (isFinalState) {
			maxQNext = 0;
		}

		int actionIndex = action.get_idAction();
		qValuesCurrent[actionIndex] = (1-learningRate)*  qValuesCurrent[actionIndex] + learningRate * (reward + gamma * maxQNext);

		QTable.put(currentState, qValuesCurrent);
	}




	@Override
	public void learn(ArrayList<TrainExample> trainExamples) {
	}
}
