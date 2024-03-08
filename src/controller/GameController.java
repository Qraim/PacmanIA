package controller;

import view.View;
import motor.*;


public class GameController {

	public PacmanGame _motor;
	private static GameController uniqueInstance;

	private GameController(PacmanGame _motor) {
		this._motor = _motor;
	}

	public static GameController getInstance(PacmanGame motor) {
		
		if(uniqueInstance == null) {
			uniqueInstance = new GameController(motor);	
		}
		
		return uniqueInstance;
		
	}
	
	public void start() { // demarre le jeu
		
		if (_motor.getEtatString() == "Init") { // si on reinitialise
			_motor.launch();
		} else {
			if (_motor.getEtatString() == "Pause") { // si on etait en pause
				_motor.stop();
			}
		}
	}

	public void restart() {
		if ((_motor.getEtatString() == "Init") || (_motor.getEtatString() == "GameOver")
				|| (_motor.getEtatString() == "Pause")) {
			_motor.init();
		}
	}

	public void pause() {
		if ((_motor.getEtatString() == "Run") || (_motor.getEtatString() == "Pause")) {
			_motor.stop();
		}
	}

	public void step() {
		if ((_motor.getEtatString() == "Pause") || (_motor.getEtatString() == "Init")) {
			_motor.step();
		}
	}

	public void speed(int speed) {
		_motor.set_speed(1000 / speed);
	}

}
