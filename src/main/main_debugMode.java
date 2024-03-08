package main;

import controller.GameController;
import motor.Maze;
import motor.PacmanGame;
import strategy.ApproximateQLearningStrategy;
import strategy.ApproximateQLearningStrategyWithNN;
import strategy.DeepQLearningStrategy;
import strategy.QLearningStrategy;
import strategy.Strategy;
import strategy.TabularQLearning;
import view.View;

public class main_debugMode {

	public static void main(String[] args) {
		

		///// Paramétrage à modifier ici : choix du niveau et de la stratégie
	
		////Choix du niveau
		int level = 0;
		
		
		////Choix de la strategie du pacman 
		
		int strategyID = 0;
		
		// 0 : Tabular Qlearning
		// 1 : Approximate Qlearning with linear model
		
		
		// Mode nightMare : les fantômes ont une stratégie A* pour se diriger vers le pacman
		boolean nightmareMode = false;
		
	
		String chemin_maze = "";

		
		if(level == 0) {
			
			chemin_maze = "layout/level0.lay";

		} else if(level == 1) {
			
			chemin_maze = "layout/level1.lay";

			
		} else if(level == 2) {
			
			chemin_maze = "layout/level2.lay";
		
		}
		
		
	    Maze _maze = null;
	    
		try {
			_maze = new Maze(chemin_maze);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		

		
		QLearningStrategy strat = null;

		double gamma = 0.95;
		double epsilon = 0.2;
		
		double learningRate;

		
		if(strategyID == 0) {
			
			learningRate = 0.1;
			
			strat = new TabularQLearning(epsilon, gamma, learningRate,  _maze.getSizeX(), _maze.getSizeY(), _maze.getNbWalls());
			
		} else if(strategyID == 1) {

			learningRate = 0.01;
			strat = new ApproximateQLearningStrategy(epsilon, gamma, learningRate,  _maze.getSizeX(), _maze.getSizeY());
			
		}
		
		
		PacmanGame _game = new PacmanGame(chemin_maze, 1000, (long) 1000);
		

		
		strat.setModeTrain(true);
		
		
		_game.initGameQLearning(strat, nightmareMode);
		
		GameController controller = GameController.getInstance(_game);
		View _view = View.getInstance(controller, _game, false);

	}
	
	
}
