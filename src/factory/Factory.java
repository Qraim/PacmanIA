package factory;

import java.util.ArrayList;

import agent.Agent;
import agent.PositionAgent;
import agent.typeAgent;
import strategy.AStarStrategy;
import strategy.RandomStrategy;
import strategy.Strategy;

public class Factory {

	private Factory() {
	}

	public static ArrayList<Agent> giveMeOnlyRandomFantom(ArrayList<PositionAgent> gostStart) {
		ArrayList<Agent> l = new ArrayList<>();

		for (PositionAgent p : gostStart)
			l.add(randomFantomFactory(p));

		return l;
	}
	

	public static ArrayList<Agent> giveMeOnlyHuntersFantom(ArrayList<PositionAgent> gostStart) {
		ArrayList<Agent> l = new ArrayList<>();

		for (PositionAgent p : gostStart)
			l.add(hunterFantomFactory(p));

		return l;
	}


	public static Agent pacmanFactory(PositionAgent position, Strategy normal, Strategy scared) {
		Agent a = new Agent(position, typeAgent.PACMAN);
		a.setBeheviorNormal(normal);
		a.setBeheviorScared(scared);
		return a;
	}

	public static Agent fantomFactory(PositionAgent position, Strategy normal, Strategy scared) {
		Agent a = new Agent(position, typeAgent.FANTOM);
		a.setBeheviorNormal(normal);
		a.setBeheviorScared(scared);
		return a;
	}

	public static Agent randomPacmanFactory(PositionAgent position) {
		return pacmanFactory(position, new RandomStrategy(), new RandomStrategy());
	}

	public static Agent randomFantomFactory(PositionAgent position) {
		return fantomFactory(position, new RandomStrategy(), new RandomStrategy());
	}

	public static Agent hunterFantomFactory(PositionAgent position) {
		return fantomFactory(position, new AStarStrategy() , new RandomStrategy());	
	}


	public static Agent player1PacmanFactory(PositionAgent position) {
		return pacmanFactory(position, null, null); // new BhvJoueur1(), new BhvJoueur1());
	}

	public static Agent playerQlearning(PositionAgent position, Strategy strat) {
		return pacmanFactory(position, strat, strat); 
	}
	
	
	
	

}
