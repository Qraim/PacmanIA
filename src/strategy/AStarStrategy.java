package strategy;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;

import agent.Agent;
import agent.AgentAction;

import agent.PositionAgent;
import agent.typeAgent;
import motor.PacmanGame;
import neuralNetwork.TrainExample;

public class AStarStrategy implements Strategy, Serializable{

	@Override
	public AgentAction play(PacmanGame game, PositionAgent positionAgent, PositionAgent objectif) {
		
		
		if(Math.random() < 0.1 ){
			

			ArrayList<AgentAction> allMoves = new ArrayList<AgentAction>();

			AgentAction actionChoosen;

			for(int i =0; i < 4; i++) {

				AgentAction action = new AgentAction(i);
				allMoves.add(action);
			}


			actionChoosen = allMoves.get((int) Math.floor(Math.random() * allMoves.size()));
				
				
			return actionChoosen;	
			
		}
		
		
		ArrayList<Node> currentNodes = new ArrayList<Node>();
		
		Node nodeStart = new Node(null,null,positionAgent.getX(),positionAgent.getY(),0,0);
		
		currentNodes.add(nodeStart);
		
		ArrayList<Integer> idNodesAlreadyExplored = new ArrayList<Integer>();
		
		boolean notfound = true;
		
		Node nodeObjectif = null;
		
		int idNodeStart= nodeStart.id;
		
		while(notfound && currentNodes.size() > 0) {
			
			Collections.sort(currentNodes, new ComparatorCoutCroissant());
					
			Node node = currentNodes.get(0);	
			currentNodes.remove(0);
				
			for(int i =0; i < 4; i++) {
				
				AgentAction action = new AgentAction(i);
				
				int newx = node.x + action.get_vx();
				int newy = node.y + action.get_vy();
				
				if(!game.getMaze().isWall(newx, newy)) {
					

					int heuristicdist = Math.abs(objectif.getX() - newx) + Math.abs(objectif.getY() - newy);
							
					Node newNode  = new Node(node, action, newx, newy, node.effectiveCost +1, heuristicdist);
					
					
					boolean isAlreadyExplored = false;
					for(int j =0; j < idNodesAlreadyExplored.size(); j++) {
						
						if(idNodesAlreadyExplored.get(j) == newNode.id) {
							isAlreadyExplored = true;
							break;
						}
						
					}
					
					if(isAlreadyExplored == false) {
		
						currentNodes.add(newNode);		
						idNodesAlreadyExplored.add(newNode.id);
										
						if(objectif.getX() == newx && objectif.getY() == newy) {
							
							nodeObjectif = newNode;
							notfound = false;
							break;
							
						}
					}
						 

				}
								
			}

		
			
		}
		

		while(nodeObjectif.parent.id != idNodeStart) {	
	     	nodeObjectif = nodeObjectif.parent;
	     	
		}
		
		return nodeObjectif.action;
			
		
	}

	
	
	
	public int manhattan(PositionAgent s, PositionAgent o) {
		return Math.abs(s.getX() - o.getX()) + Math.abs(s.getY() - o.getY());
	}
	
	
	public class Node {
		
		public int x;
		public int y;
		
		public int heuristicCost;
		public int effectiveCost;
		
		public int globalCost;
		
		public int id;
		
		public Node parent;
		public AgentAction action;
		
		public Node(Node node, AgentAction action, int x,int y, int effectiveCost, int heuristicCost) {
			
			this.parent = node;
			this.action = action;
			this.x = x;
			this.y = y;
			this.effectiveCost = effectiveCost;
			this.heuristicCost = heuristicCost;
			
			this.globalCost = effectiveCost + heuristicCost;
			this.id = x*100+ y;
		}
		
	}
	
	
	public class ComparatorCoutCroissant implements Comparator<Node> {
		@Override
		public int compare(Node a, Node b) {
			return a.globalCost - b.globalCost;
		}
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
