package strategy;

import java.util.ArrayList;

import agent.AgentAction;
import agent.PositionAgent;
import agent.typeAgent;
import motor.Maze;
import motor.PacmanGame;
import neuralNetwork.TrainExample;

public interface Strategy {
	

	
	public AgentAction play(PacmanGame state, PositionAgent positionAgent, PositionAgent objectif);
	

	public void update(PacmanGame state, PacmanGame nextState, AgentAction action, double reward, boolean isFinalState);

	public boolean isModeTrain();

	

}
