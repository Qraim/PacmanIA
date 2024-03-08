package strategy;

import java.io.Serializable;
import java.util.ArrayList;

import agent.AgentAction;
import agent.PositionAgent;
import motor.PacmanGame;
import neuralNetwork.TrainExample;


public abstract class QLearningStrategy implements Strategy{

	protected double base_epsilon;
	protected double current_epsilon;
	protected double gamma;
	protected double learningRate;
	
	private boolean modeTrain;
	
	
	
	public ArrayList<TrainExample> trainExamples  = new ArrayList<TrainExample>();
	
	int sizeMazeX;
	int sizeMazeY;
	
	public QLearningStrategy(double base_epsilon, double gamma, double learningRate, int sizeMazeX, int sizeMazeY) {
		
		this.base_epsilon = base_epsilon;
		this.current_epsilon = base_epsilon;
		this.gamma = gamma;
		this.learningRate = learningRate;
		
		this.sizeMazeX = sizeMazeX;
		this.sizeMazeY = sizeMazeY;
		
		
	}
	
	
	public AgentAction play(PacmanGame game, PositionAgent positionAgent, PositionAgent objectif) {
		
		return this.chooseAction(game);
	}
	

	
	public abstract AgentAction chooseAction(PacmanGame state);	
	
	public abstract void learn(ArrayList<TrainExample> trainExamples);
	
	public abstract void update(PacmanGame state, PacmanGame nextState, AgentAction action, double reward, boolean isFinalState);
	


	@Override
	public boolean isModeTrain() {
		
		return this.modeTrain;
	}


	public void setModeTrain(boolean modeTrain) {
		
		if(modeTrain) {
			this.current_epsilon  = this.base_epsilon;
		} else {
			this.current_epsilon  = 0;
		}
			
		this.modeTrain = modeTrain;
	}
	
	

}
