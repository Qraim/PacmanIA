package strategy;

import java.util.ArrayList;
import java.util.Random;

import agent.Agent;
import agent.AgentAction;
import agent.PositionAgent;
import motor.Maze;
import motor.PacmanGame;
import neuralNetwork.TrainExample;

public class ApproximateQLearningStrategy extends QLearningStrategy {
	private double[] weights;
	private Random rand = new Random();

	public ApproximateQLearningStrategy(double epsilon, double gamma, double alpha, int sizeMazeX, int sizeMazeY) {
		super(epsilon, gamma, alpha, sizeMazeX, sizeMazeY);
		int numberOfFeatures = 6;
		this.weights = new double[numberOfFeatures];

		// Initialisation des poids avec des valeurs aléatoires entre -1 et 1
		for (int i = 0; i < this.weights.length; i++) {
			this.weights[i] = rand.nextDouble() * 2 - 1;
		}
	}

	@Override
	public AgentAction chooseAction(PacmanGame state) {
		ArrayList<AgentAction> legalActions = new ArrayList<AgentAction>(); // Actions légales possibles
		Maze maze = state.getMaze(); // Le labyrinthe du jeu
		AgentAction actionChoosen = new AgentAction(0); // Action choisie

		// Vérification des actions légales (ne menant pas à un mur)
		for(int i =0; i < 4; i++) {
			AgentAction action = new AgentAction(i);
			if(!maze.isWall(state.pacman.get_position().getX() + action.get_vx(),
					state.pacman.get_position().getY() + action.get_vy())) {
				legalActions.add(action);
			}
		}
		// Choix aléatoire d'une action si un nombre aléatoire est inférieur à epsilon (exploration)
		if(Math.random() < this.current_epsilon){
			actionChoosen = legalActions.get((int) Math.floor(Math.random() * legalActions.size()));
		} else {
			// Sinon, choisir l'action maximisant la valeur Q estimée (exploitation)
			double maxQvalue = -9999;

			int trouve = 1;

			for(AgentAction action : legalActions) {

				double[] features = extractFeatures(state, action);
				double qValue = perceptron(weights, features);

				// Mise à jour de l'action choisie si la valeur Q est supérieure à la valeur Q maximale actuelle
				if(qValue > maxQvalue) {

					maxQvalue = qValue;
					actionChoosen = action;
					trouve = 1;

				} else if(qValue == maxQvalue) {
					// En cas d'égalité, choisir aléatoirement entre les actions équivalentes
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


		double targetQ;

		// Si c'est un état final, la valeur cible Q est simplement la récompense
		if(isFinalState) {

			targetQ = reward;

		} else {

			// Sinon, calculer la valeur cible Q avec la récompense et la meilleure valeur Q future estimée
			double maxQnext = getMaxQNext(nextState);

			targetQ = reward + this.gamma*maxQnext;

		}

		// Mise à jour des poids avec la règle de mise à jour de l'apprentissage par renforcement
		double[] features = extractFeatures(state, action); // Extraction des caractéristiques
		double qValue = perceptron(weights, features); // Calcul de la valeur Q actuelle

		for(int i =0; i < this.weights.length; i ++) {

			// Mise à jour des poids avec le taux d'apprentissage et l'erreur (différence entre la valeur Q cible et actuelle)
			this.weights[i] = this.weights[i] - 2*this.learningRate*features[i]*(qValue - targetQ);

		}


	}
	public double perceptron(double[] weights, double[] features) {

		double results = 0;

		for(int i =0; i < weights.length; i++) {
			results += weights[i]*features[i];
		}

		return results;
	}

	public double getMaxQNext(PacmanGame game) {
		PositionAgent nextPos = game.pacman._position;
		Maze maze = game.getMaze();
		double maxQvalue = -99999;
		for(int i =0; i < 4; i++) {
			AgentAction action = new AgentAction(i);
			if(!maze.isWall(nextPos.getX() + action.get_vx(),
					nextPos.getY() + action.get_vy())) {

				double[] features = extractFeatures(game, action);
				double qValue = perceptron(weights, features);

				// Mise à jour de la valeur Q maximale si nécessaire
				if(qValue > maxQvalue) {

					maxQvalue = qValue;

				}
			}
		}
		return maxQvalue;
	}

	// Méthode pour extraire les caractéristiques d'un état et d'une action
	private double[] extractFeatures(PacmanGame state, AgentAction action) {


		double[] features = new double[6];

		features[0] = 1;

		Maze maze = state.getMaze();

		int x = state.pacman._position.getX();
		int y = state.pacman._position.getY();

		int new_x = x + action.get_vx();
		int new_y = y + action.get_vy();

		/*
		 * 1 : Si mange une pacgomme (0 ou 1)
		 * 2: Si mange une capsule (0 ou 1)
		 * 3 : Nombre de fantômes autour de la case cible quand Pacman est invulnérable
		 * 4: Nombre de fantômes autour de la case cible quand Pacman est vulnérable
		 * 5: Nombre de coups avant prochain objectif
		 * 6: Nombre de tours restants à l'invincibilité
		 */

		if(maze.isFood(new_x, y + action.get_vy())) {
			features[1] = 1; // Si mange une pacgomme
		}

		if(maze.isCapsule(new_x, new_y)) {
			features[2]=1; // Si mange une capsule
		}
		if(state.getNbTourInvincible()>1) { // Nombre de fantômes quand invincible
			features[3]=countGhostsAround(new_x,new_y,state);
		}else{ // Nombre de fantômes quand vulnérable
			features[4]=countGhostsAround(new_x,new_y,state);
		}
		features[5]=nbCoupsProchainePacgomme(state,new_x,new_y); // Nombre de coups avant prochain objectif

		return features;

	}


	@Override
	public void learn(ArrayList<TrainExample> trainExamples) {
		// pas utilisé si pas de reseaux neuronaux
	}
}
