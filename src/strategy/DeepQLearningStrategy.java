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
	public void update(PacmanGame state, PacmanGame nextState, AgentAction action, double reward,
					   boolean isFinalState) {

		double[] X = getEncodedState(state);
		double[] Y = this.nn.predict(X);

		double maxQvalue_nextState=-999999;
		if(isFinalState){
			maxQvalue_nextState = 0;
		}else{
			maxQvalue_nextState = getMaxQNext(nextState);
		}

		Y[action.get_idAction()] = reward + gamma*maxQvalue_nextState;

		TrainExample trainExample = new TrainExample(X,Y);
		this.trainExamples.add(trainExample);

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
			int trouve=1;

			for(AgentAction action : legalActions) {
				double[] encodedState=getEncodedState(state);
				double[] output = this.nn.predict(encodedState);
				if(maxQvalue<output[action.get_idAction()]) {
					maxQvalue=output[action.get_idAction()];
					actionChoosen=action;
					trouve=1;
				}
				else if(maxQvalue==output[action.get_idAction()]) {
					trouve+=1;
					if(Math.floor(trouve*Math.random())== 0) {
						maxQvalue = output[action.get_idAction()];
						actionChoosen = action;
					}
				}
			}

		}

		return actionChoosen;
	}


	@Override
	public void learn(ArrayList<TrainExample> trainExamples) {
		nn.fit(trainExamples, nEpochs, batchSize,this.learningRate);

	}

	public double[] getEncodedState(PacmanGame game){
		double[] result=new double[sizeState];
		int iWall=0;
		int iFood=range*range;
		int iGhost=range*range*2;

		int x = game.pacman._position.getX();
		int y = game.pacman._position.getY();
		Maze maze=game.getMaze();

		for(int i=x-(range/2);i<=x+(range/2);i++) {
			for(int j=y-(range/2);j<=x+(range/2);j++) {

				//ajout des murs
				if(maze.isWall(i, j)) {
					result[iWall]=1;
					result[iFood]=0;
					result[iGhost]=0;
				}else{
					result[iWall]=0;

					//ajout des foods & capsules
					double isFood = maze.isFood(i, j)? (double)1 : (double)0;
					result[iFood]=maze.isCapsule(i, j)? -1 : isFood;

					//ajout des fantômes
					boolean ghostSeen=false;
					for(PositionAgent gPos:game.getPostionFantom()) {
						if(gPos.getX()==i && gPos.getY()==j) {
							ghostSeen=true;
							if(game.isGhostsScarred()) result[iGhost]=-1;
							else result[iGhost]=1;
							break;
						}
					}
					if(!ghostSeen)result[iGhost]=0;
				}
				iWall++;
				iFood++;
				iGhost++;
			}
		}
		String affichage=" result :";
		for(int i=0;i<result.length;i++) {
			affichage+=result[i]+" ";
		}
		System.out.println(affichage);
		return result;
	}

	public double getMaxQNext(PacmanGame game){
		double maxQvalue = -99999;
		double[] nextQValue = this.nn.predict(getEncodedState(game));

		for(int i =0; i < 4; i++) {
			if(nextQValue[i]>maxQvalue) maxQvalue=nextQValue[i];
		}

		return maxQvalue;
	}


}