package strategy;

import java.util.ArrayList;
import java.util.List;

import agent.Agent;
import agent.AgentAction;

import agent.PositionAgent;
import agent.typeAgent;
import motor.Maze;
import motor.PacmanGame;
import neuralNetwork.NeuralNetWorkDL4J;

import neuralNetwork.TrainExample;

import java.util.Random;


public class DeepQLearningStrategy extends QLearningStrategy {

	int nEpochs;
	int batchSize;
	
	int range;
	
	NeuralNetWorkDL4J nn;
	int sizeState;
	
	boolean modeAllMaze;
		
	
	public DeepQLearningStrategy(double epsilon, double gamma, double alpha, int range, int nEpochs, int batchSize,  int sizeMazeX, int sizeMazeY, boolean modeAllMaze, int nbWalls) {
		
		
		super(epsilon, gamma, alpha, sizeMazeX, sizeMazeY);

		
		this.modeAllMaze = modeAllMaze;
		
		System.out.println("nbWalls : " + nbWalls);
		
		if(modeAllMaze) {
			
			this.sizeState = (sizeMazeX)*(sizeMazeY)*4 - nbWalls;
			
		} else {
			this.sizeState = range*range*4;
		}
		
		System.out.println("Size entry neural network : " + this.sizeState);
		
		this.nn = new NeuralNetWorkDL4J(alpha, 0, sizeState, 4);
		
		this.nEpochs = nEpochs;
		this.batchSize = batchSize;
		
		this.range = range;
		
		
		
	}
	

	@Override
	public AgentAction chooseAction(PacmanGame state) {

		return null;

	}


	
	@Override
	public void update(PacmanGame state, PacmanGame nextState, AgentAction action, double reward, boolean isFinalState) {



		
	}

	
	
	
	public void learn(ArrayList<TrainExample> trainExamples) {
		
		nn.fit(trainExamples, nEpochs, batchSize, learningRate);
	}
	
	
}
