package strategy;

import agent.Agent;
import agent.AgentAction;
import motor.PacmanGame;
import neuralNetwork.TrainExample;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.ArrayList;
import java.util.Random;

public class ApproximateQLearningStrategyWithNN extends QLearningStrategy {

	private MultiLayerNetwork nn;
	private int nEpochs;
	private int batchSize;

	public ApproximateQLearningStrategyWithNN(double epsilon, double gamma, double alpha, int nEpochs, int batchSize, int sizeMazeX, int sizeMazeY) {
		super(epsilon, gamma, alpha, sizeMazeX, sizeMazeY);

		this.nEpochs = nEpochs;
		this.batchSize = batchSize;

		int numInputs = 7; // Nombre de caractéristiques extraites
		int numOutputs = 4; // Nombre d'actions possibles
		int numHiddenNodes = 10; // Exemple arbitraire de taille pour la couche cachée

		org.deeplearning4j.nn.conf.MultiLayerConfiguration nnConfig = new NeuralNetConfiguration.Builder()
				.updater(new Adam(alpha))
				.list()
				.layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
						.activation(Activation.RELU)
						.build())
				.layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
						.activation(Activation.IDENTITY)
						.nIn(numHiddenNodes).nOut(numOutputs)
						.build())
				.build();

		this.nn = new MultiLayerNetwork(nnConfig);
		this.nn.init();
	}

	@Override
	public AgentAction chooseAction(PacmanGame state) {
		INDArray features = Nd4j.create(extractFeatures(state, new AgentAction(0))); // Ajustez cela pour chaque action possible si nécessaire
		INDArray output = nn.output(features);
		int actionIndex = Nd4j.argMax(output, 1).getInt(0);
		return new AgentAction(actionIndex);
	}



	@Override
	public void update(PacmanGame state, PacmanGame nextState, AgentAction action, double reward, boolean isFinalState) {
		INDArray currentStateFeatures = Nd4j.create(extractFeatures(state, action)).reshape(1, -1); // Ajustement pour garantir que les données sont au format matrice.
		INDArray currentQValues = nn.output(currentStateFeatures);

		double maxNextQ = isFinalState ? 0 : Double.NEGATIVE_INFINITY;
		for (int a = 0; a < AgentAction.NUMBER_OF_ACTIONS; a++) {
			AgentAction nextAction = new AgentAction(a);
			INDArray nextStateFeatures = Nd4j.create(extractFeatures(nextState, nextAction)).reshape(1, -1); // Ajustement similaire pour les caractéristiques de l'état suivant.
			INDArray nextQValues = nn.output(nextStateFeatures);
			maxNextQ = Math.max(maxNextQ, nextQValues.maxNumber().doubleValue());
		}

		int actionIndex = action.get_idAction();
		double targetQ = reward + gamma * maxNextQ;
		currentQValues.putScalar(new int[]{0, actionIndex}, targetQ);

		nn.fit(currentStateFeatures, currentQValues); // Assurez-vous que les deux INDArrays sont bien dimensionnés.
	}



	@Override
	public void learn(ArrayList<TrainExample> trainExamples) {
		// L'apprentissage du réseau à partir des exemples collectés doit être implémenté ici
	}
	private double[][] extractFeatures(PacmanGame state, AgentAction action) {
		double[][] features = new double[1][7]; // Créez un tableau 2D avec 1 ligne et 7 colonnes

		int nextX = state.getPacmanPosition().getX() + action.get_vx();
		int nextY = state.getPacmanPosition().getY() + action.get_vy();

		features[0][0] = calculateDistanceToNearestGhost(state, nextX, nextY);
		features[0][1] = state.getNbFood();
		features[0][2] = state.isCapsuleAtPosition(nextX, nextY) ? 1 : 0;
		features[0][3] = state.isWallAtPosition(nextX, nextY) ? 1 : 0;
		features[0][4] = state.isGhostsScarred() ? 1 : 0;
		features[0][5] = state.isLegalMovePacman(action) ? 1 : 0; // Assurez-vous que cette vérification est cohérente avec votre implémentation
		features[0][6] = state.isGumAtPosition(nextX, nextY) ? 1 : 0;

		return features;
	}


	private double calculateDistanceToNearestGhost(PacmanGame state, int nextX, int nextY) {
		double nearestDistance = Double.MAX_VALUE;

		for ( Agent ghost : state.get_agentsFantom()) {
			int ghostX = ghost.get_position().getX();
			int ghostY = ghost.get_position().getY();

			double distance = Math.abs(nextX - ghostX) + Math.abs(nextY - ghostY);

			if (distance < nearestDistance) {
				nearestDistance = distance;
			}
		}
		return nearestDistance;
	}


}
