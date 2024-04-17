package strategy;

import agent.Agent;
import agent.AgentAction;
import agent.PositionAgent;
import motor.Maze;
import motor.PacmanGame;
import neuralNetwork.NeuralNetWorkDL4J;
import neuralNetwork.TrainExample;


import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

public class ApproximateQLearningStrategyWithNN extends QLearningStrategy {

	private NeuralNetWorkDL4J nn;
	private int nEpochs;
	private int batchSize;

	public ApproximateQLearningStrategyWithNN(double epsilon, double gamma, double alpha, int nEpochs, int batchSize, int sizeMazeX, int sizeMazeY) {
		super(epsilon, gamma, alpha, sizeMazeX, sizeMazeY);

		this.nEpochs = nEpochs;
		this.batchSize = batchSize;

		this.nEpochs = nEpochs;
		this.batchSize = batchSize;


		this.nn=new NeuralNetWorkDL4J(alpha,0,7,1);
	}

	@Override
	public AgentAction chooseAction(PacmanGame state) {
		ArrayList<AgentAction> legalActions = new ArrayList<AgentAction>();

		Maze maze = state.getMaze();

		AgentAction actionChoosen = new AgentAction(0);


		for(int i =0; i < 4; i++) {

			AgentAction action = new AgentAction(i);

			if(!maze.isWall(state.pacman.get_position().getX() + action.get_vx(),
					state.pacman.get_position().getY() + action.get_vy())) {

				legalActions.add(action);
			}

		}


		if(Math.random() < this.current_epsilon){

			actionChoosen = legalActions.get((int) Math.floor(Math.random() * legalActions.size()));

		} else {

			double maxQvalue = -9999;

			int trouve = 1;

			for(AgentAction action : legalActions) {

				double[] features = extractFeatures(state, action);
				double qValue = this.nn.predict(features)[0];

				if(qValue > maxQvalue) {
					maxQvalue = qValue;
					actionChoosen = action;
					trouve = 1;
				} else if(qValue == maxQvalue) {
					trouve += 1;

					if(Math.floor(trouve*Math.random())== 0) {
						maxQvalue = qValue;
						actionChoosen = action;
					}
				}
			}
		}
		return actionChoosen;
	}

	@Override
	public void update(PacmanGame state, PacmanGame nextState, AgentAction action, double reward, boolean isFinalState) {
		double[] targetQ=new double[1];

		if(isFinalState) {

			targetQ[0] = reward;

		} else {

			double maxQnext = getMaxQNext(nextState);
			targetQ[0] = reward + this.gamma*maxQnext;

		}

		double[] features = extractFeatures(state, action);
		this.trainExamples.add(new TrainExample(features,targetQ));
	}


	public double getMaxQNext(PacmanGame game) {

		PositionAgent nextPos = game.pacman._position;
		Maze maze = game.getMaze();

		double maxQvalue = -99999;

		for(int i =0; i < 4; i++) {


			AgentAction action = new AgentAction(i);
			if(!maze.isWall(nextPos.getX() + action.get_vx(),
					nextPos.getY() + action.get_vy())) {

				double[] features = extractFeatures( game, action);
				double qValue = this.nn.predict(features)[0];;

				if(qValue > maxQvalue) {

					maxQvalue = qValue;

				}

			}

		}

		return maxQvalue;

	}


	@Override
	public void learn(ArrayList<TrainExample> trainExamples) {
		if (trainExamples.isEmpty()) {
			System.out.println("No training examples available, skipping learning phase.");
			return;
		}
		nn.fit(trainExamples, nEpochs, batchSize,this.learningRate);
	}

	private double[] extractFeatures(PacmanGame state, AgentAction action) {


		double[] features = new double[7];

		features[0] = 1;

		Maze maze = state.getMaze();

		int x = state.pacman._position.getX();
		int y = state.pacman._position.getY();

		int new_x = x + action.get_vx();
		int new_y = y + action.get_vy();


		if(maze.isFood(new_x, y + action.get_vy())) {
			features[1] = 1;
		}

		if(maze.isCapsule(new_x, new_y)) {
			features[2]=1;
		}

		if(state.getNbTourInvincible()>1) { // nb fantômes quand invincible
			features[3]=countGhostsAround(new_x,new_y,state);
		}else{ //nb fantômes quand vulnérable
			features[4]=countGhostsAround(new_x,new_y,state);
		}

		features[5]=nbCoupsProchainePacgomme(state,new_x,new_y);

		features[6]=state.getNbTourInvincible();

		//features[7]=distMoyennneFantomes(state,new_x,new_y);

		return features;


	}

}
