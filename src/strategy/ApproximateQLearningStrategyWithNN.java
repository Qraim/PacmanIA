package strategy;

import java.util.ArrayList;
import java.util.List;

import agent.Agent;
import agent.AgentAction;

import agent.PositionAgent;

import motor.PacmanGame;
import neuralNetwork.NeuralNetWorkDL4J;

import neuralNetwork.TrainExample;



public class ApproximateQLearningStrategyWithNN extends QLearningStrategy {

	int d;
	NeuralNetWorkDL4J nn;

	int nEpochs;
	int batchSize;


	public ApproximateQLearningStrategyWithNN(double epsilon, double gamma, double learningRate,  int nEpochs, int batchSize, int sizeMazeX , int sizeMazeY) {

		super(epsilon, gamma, learningRate, sizeMazeX, sizeMazeY);

		this.nEpochs = nEpochs;
		this.batchSize = batchSize;

		
		// A compléter


	}

	@Override
	public AgentAction chooseAction(PacmanGame state) {

	
		return null;

	}



	

	@Override
	public void update(PacmanGame state, PacmanGame nextState, AgentAction action, double reward, boolean isFinalState) {


		
	}


	@Override
	public void learn(ArrayList<TrainExample> trainExamples) {

		nn.fit(trainExamples, this.nEpochs, this.batchSize, this.learningRate);

	}







}
