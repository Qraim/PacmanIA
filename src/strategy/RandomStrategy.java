package strategy;

import java.io.Serializable;
import java.util.ArrayList;

import agent.AgentAction;
import agent.PositionAgent;
import motor.PacmanGame;


public class RandomStrategy implements Strategy, Serializable{

	@Override
	public AgentAction play(PacmanGame state, PositionAgent positionAgent, PositionAgent objectif) {
		

		ArrayList<AgentAction> legalMoves = new ArrayList<AgentAction>();

		AgentAction actionChoosen;

		for(int i =0; i < 4; i++) {

			AgentAction action = new AgentAction(i);

			if(state.isLegalMove(state.pacman, action)) {
				legalMoves.add(action);
			}

		}


		actionChoosen = legalMoves.get((int) Math.floor(Math.random() * legalMoves.size()));
			
			
		return actionChoosen;
		
	}

	@Override
	public void update(PacmanGame state, PacmanGame nextState, AgentAction action, double reward,
			boolean isFinalState) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public boolean isModeTrain() {
		// TODO Auto-generated method stub
		return false;
	}



}
