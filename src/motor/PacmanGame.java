package motor;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.apache.commons.lang3.SerializationUtils;

import agent.Agent;
import agent.AgentAction;
import agent.PositionAgent;
import agent.typeAgent;
import factory.Factory;
import strategy.Strategy;


public class PacmanGame extends Game  {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private static final int TOUR_MAX_INVINCIBLE = 10;
	
	private static final int POINT_GUM = 25;
	private static final int POINT_FANTOM = 50;
	private static final int POINT_CAPSULE = 50;
	private static final int POINT_DEAD = -100;
	private static final int POINT_WIN = 100;
	
	
	private Maze _maze;
	private Maze _originalMaze;
	
	public Agent pacman;
	private List<Agent> _agentsFantom;

	private int score;

	private boolean ghostsScarred;
	private int nb_tour_invincible;

	private int nbCapsule;
	private int nbFood;
	
	boolean nightmareMode = false;
	
	public PacmanGame(String chemin_maze, int maximum_laps, long speed) {
		super(maximum_laps, speed);
		
	    this._maze = null;
	   
	    this.score = 0;
	    
		try {
			_maze = new Maze(chemin_maze);
		} catch (Exception e) {
			e.printStackTrace();
		}

		this._originalMaze = SerializationUtils.clone(this._maze);
		
	}

	public PacmanGame(int maximum_laps) {
		super(maximum_laps);
	}

	
	public void initGameQLearning(Strategy behavior, boolean nightmareMode ) {
		
		super.initGame();

		this.nightmareMode = nightmareMode;
		
		if(nightmareMode) {
			_agentsFantom = Factory.giveMeOnlyHuntersFantom(_maze.getGhosts_start());
		} else {
			_agentsFantom = Factory.giveMeOnlyRandomFantom(_maze.getGhosts_start());
		}
		
		pacman = Factory.playerQlearning(_maze.getPacman_start().get(0), behavior);
		
		ghostsScarred = false;
		nb_tour_invincible = 0;
		score = 0;

		notifyObservers();
		
		nbCapsule = countCapsules(_maze);
		nbFood = countFoods(_maze);
		
	}
	
	
	
	public int countCapsules(Maze maze) {
		
		int nb = 0;
		
		for(int i =0; i < maze.getSizeX(); i++) {
			
			for(int j =0; j < maze.getSizeY(); j++) {
				
				if(maze.isCapsule(i, j)) {
					
					nb += 1;
				}
				
			}
		}
		
		return nb;
	}
	
	
	public int countFoods(Maze maze) {
		
		int nb = 0;
		
		for(int i =0; i < maze.getSizeX(); i++) {
			
			for(int j =0; j < maze.getSizeY(); j++) {
				
				if(maze.isFood(i, j)) {
					
					nb += 1;
				}
				
			}
		}
		
		return nb;
	}
	
	
	public boolean isLegalMove(Agent agent, AgentAction action) {
		if (agent.is_alive()) {
			return !_maze.isWall(agent.get_position().getX() + action.get_vx(),
					agent.get_position().getY() + action.get_vy());
		}
		return false;
	}

	public boolean isLegalMovePacman(AgentAction action) {
		if (pacman.is_alive()) {
			return !_maze.isWall(pacman.get_position().getX() + action.get_vx(),
					pacman.get_position().getY() + action.get_vy());
		}
		return false;
	}
	
	
	public boolean isGhostAtPosition(int x, int y) {
		
		for (Agent fantom : _agentsFantom) {
			
			if(fantom.get_position().getX() == x && fantom.get_position().getY() == y) {
				
				return true;
			}
			
		}
		
		return false;
	}
	
	public boolean isPacmanAtPosition(int x, int y) {
		

		if(pacman.get_position().getX() == x && pacman.get_position().getY() == y) {
				
			return true;
			
		} else {
			return false;
		}
			

		
	}
	
	
	
	public boolean isWallAtPosition(int x, int y) {
		
		return this._maze.isWall(x, y);
	}

	public boolean isGumAtPosition(int x, int y) {
		
		return this._maze.isFood(x, y);
	}
	
	public boolean isCapsuleAtPosition(int x, int y) {
		
		return this._maze.isCapsule(x, y);
	}
	
	
	
	public PositionAgent getPacmanPosition() {
		
		return this.pacman.get_position();
		
	}
	
	public int getPacmanX() {
		
		return this.pacman.get_position().getX();
		
	}	

	
	public int getPacmanY() {
		
		return this.pacman.get_position().getY();
		
	}	
	
	
	public ArrayList<AgentAction> getLegalPacmanActions(){
		
		ArrayList<AgentAction> legalMoves = new ArrayList<AgentAction>();



		for(int i =0; i < 4; i++) {

			AgentAction action = new AgentAction(i);

			if(this.isLegalMovePacman(action)) {
				legalMoves.add(action);
			}

		}
		
		return legalMoves;
	}
	
	
	public void moveAgent(Agent agent, AgentAction action) {

		int dirX = agent.get_position().getX() + action.get_vx();
		int dirY = agent.get_position().getY() + action.get_vy();

		agent.set_position(new PositionAgent(dirX, dirY, action.get_idAction()));

		if (agent.get_type() == typeAgent.PACMAN) { 
			if (_maze.isCapsule(dirX, dirY)) {
				eatCapsule(dirX, dirY);
			    this.nbCapsule -= 1;
			}  
			if (_maze.isFood(dirX, dirY)) {
				eatGum(dirX, dirY); 
				this.nbFood -= 1;
			}
		}


	}

	public boolean samePosition(Agent a1, Agent a2) {
		return (a1.get_position().getX() == a2.get_position().getX())
				&& (a1.get_position().getY() == a2.get_position().getY());
	}

	
	public void eatCapsule(int x, int y) {
		// on commence l'invincibilité
		nb_tour_invincible = TOUR_MAX_INVINCIBLE;
		score += POINT_CAPSULE;
		
		setGhostsScarred(true);

		// on mange la capsule (enleve du terrain)
		_maze.setCapsule(x, y, false);
	}

	public void eatGum(int x, int y) {
		
		score += POINT_GUM;
		_maze.setFood(x, y, false);
	}

	
	@Override
	public void reinitializeGameStat() {

		this._maze = SerializationUtils.clone(this._originalMaze);
		
		ghostsScarred = false;
		nb_tour_invincible = 0;
		score = 0;


		if(nightmareMode) {
			_agentsFantom = Factory.giveMeOnlyHuntersFantom(_maze.getGhosts_start());
		} else {
			_agentsFantom = Factory.giveMeOnlyRandomFantom(_maze.getGhosts_start());
		}
		
		
		pacman.set_alive(true);
		pacman.set_position(pacman.get_first_position());
		
		nbCapsule = countCapsules(_maze);
		nbFood = countFoods(_maze);
		
		
		notifyObservers();
	}


	@Override
	public void takeTurn() {

			
		if (isGhostsScarred()) {
			if (nb_tour_invincible == 0) {
				setGhostsScarred(false);
			} else {
				nb_tour_invincible--;
			}
		}

		double reward =  - this.score;

		AgentAction a;
		
		PacmanGame state = SerializationUtils.clone(this);
		
		a = pacman.play(this, ghostsScarred, null);
		if (isLegalMove(pacman, a)) {
			moveAgent(pacman, a);
		}


		if(_agentsFantom.size() > 0) {
			
			Iterator<Agent> iterator = _agentsFantom.iterator(); 
			while (iterator.hasNext()) {
				
				Agent fantom = (Agent)iterator.next();
				
				boolean isRemoved = false;
						
				if(samePosition(pacman,fantom )) {
					
					if (isGhostsScarred()) {
						
						score += POINT_FANTOM;
						iterator.remove();
						isRemoved = true;
						
					} else {
						
						score += PacmanGame.POINT_DEAD;
						pacman.set_alive(false);
						
					}
				}
				
				if(!isRemoved && pacman.is_alive()) {
					
					AgentAction aa = fantom.play(this, ghostsScarred, pacman.get_position());
					if (isLegalMove(fantom, aa)) {
						moveAgent(fantom, aa);
					}
					
					if(samePosition(pacman,fantom)) {
					
						if (isGhostsScarred()) {
							score += POINT_FANTOM;						
							iterator.remove();
						} else {
							score += PacmanGame.POINT_DEAD;
							pacman.set_alive(false);
						}
					}
				}
				
				
			}
		}
		
	
		
		notifyObservers();
		
		boolean isFinalState = checkIfFinalState();
		
		
		reward += this.score;
		
		if(pacman.get_behavior_normal().isModeTrain()) {	
			pacman.update(state, this,  a, reward, isFinalState);
		}
		
		if(isFinalState) {
			gameOver();
		}
	}

	public boolean checkIfFinalState() {

		boolean fin_gum = false; 
		boolean fin_pacman = false; 
		
		if(nbCapsule == 0 && nbFood == 0) {
			fin_gum = true;
		} else {
			fin_gum = false;
		}


		if (pacman.is_alive()) {
			fin_pacman = false;
		} else {
			fin_pacman = true;
		}
		
		if (fin_gum) {
			score += PacmanGame.POINT_WIN;		
		}
		
		return (fin_gum || fin_pacman);
	}


	@Override
	public void gameOver() {
		super.gameOver();
	
	}



	public List<Agent> get_agentsFantom() {
		return _agentsFantom;
	}

	public void set_agentsFantom(List<Agent> _agentsFantom) {
		this._agentsFantom = _agentsFantom;
	}

	public ArrayList<PositionAgent> getPostionPacman() {
		ArrayList<PositionAgent> pa = new ArrayList<PositionAgent>();
	
		pa.add(pacman.get_position());
		
		return pa;
	}

	public ArrayList<PositionAgent> getPostionFantom() {
		ArrayList<PositionAgent> pa = new ArrayList<PositionAgent>();
		for (Agent agent : _agentsFantom) {
			pa.add(agent.get_position());
		}
		return pa;
	}

	public Maze getMaze() {
		return _maze;
	}

	public void setMaze(Maze _maze) {
		this._maze = _maze;
		
		initGame();
	}

	public boolean isGhostsScarred() {
		return ghostsScarred;
	}

	public void setGhostsScarred(boolean ghostsScarred) {
		this.ghostsScarred = ghostsScarred;
	}

	public int getScore() {
		return this.score;
	}

	public int getNbcapsule() {
		return nbCapsule;
	}
	
	public int getNbFood() {
		
		return this.nbFood;
	}

	public int getNbTourInvincible() {
		return nb_tour_invincible;
	}


}