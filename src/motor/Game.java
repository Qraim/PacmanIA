package motor;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import observator_pattern.*;

public abstract class Game implements Runnable, Observable, Serializable {

	private int _counter_nb_laps;
	private int _maximum_laps;
	private boolean _isRunning;
	private long _speed;
	private State _etatCourant;
	transient Thread _thread;
	private transient List<Observer> _observers;

	public Game(int maximum_laps, long speed) {
		_maximum_laps = maximum_laps;
		_speed = speed;
		_observers = new ArrayList<Observer>();
	}

	public Game(int maximum_laps) {
		_maximum_laps = maximum_laps;
		_speed = 1000;
		initGame();
	}

	public void initGame() { 
		_counter_nb_laps 	= 0;
		_isRunning 			= true;
		_etatCourant 		= State.Start;

	}

	public void launch() {
		_isRunning = true;
		_thread = new Thread(this);
		_thread.start();
	}

	public void init() {
		_counter_nb_laps 	= 0;
		_isRunning 			= false;
		_etatCourant 		= State.Start;
		reinitializeGameStat();
		notifyObservers();

	}

	public abstract void reinitializeGameStat();
	

	public void step() {
		_counter_nb_laps++;
		if (!isMaxLapsReached()) {
			takeTurn();
		} else {
			gameOver();
			_etatCourant = State.GameOver;
		}
		notifyObservers();
	}

	public abstract void takeTurn();

	public void gameOver(){
		_isRunning 		= false;
		_etatCourant 	= State.GameOver;
	}

	public void victoire(){
		_isRunning 		= false;
		_etatCourant 	= State.Victoire;
	}
	
	public boolean isMaxLapsReached() {
		return (_counter_nb_laps > _maximum_laps);
	}

	public void run() {
		while ((!isMaxLapsReached()) && _isRunning) {
			_etatCourant = State.Run;
			
			step();
			
			if(_speed != -1) {
				
				try {
					Thread.sleep(_speed);
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
			}
		
		}
	}

	public void join() throws InterruptedException {
		_thread.join();
	}
	
	public void stop() {
		if (_isRunning) {
			_isRunning 		= false;
			_etatCourant 	= State.Pause;
		} else {
			_isRunning 		= true;
			_etatCourant 	= State.Run;
			launch();
		}
		notifyObservers();
	}

	/**//**//**//**//**//**//**//**//**//**//**//**/// Observer pattern

	@Override
	public void addObserver(Observer o) {
		_observers.add(o);
	}

	@Override
	public void deleteObserver(Observer o) {
		_observers.remove(o);
	}

	@Override
	public void notifyObservers() {
		for (Observer o : _observers)
			o.update();
	}

	/**//**//**//**//**//**//**//**//**//**//**//**/// Getters
	public int get_counter_nb_laps() {
		return _counter_nb_laps;
	}

	public int get_maximum_laps() {
		return _maximum_laps;
	}

	public boolean isRunning() {
		return _isRunning;
	}

	public void setIsRunning(boolean bool) {
		_isRunning = bool;
	}

	public long get_speed() {
		return _speed;
	}

	public void set_speed(long speed) {
		_speed = speed;
	}

	public State getEtatCourant() {
		return _etatCourant;
	}

	public String getEtatString() {
		String etat;
		switch (_etatCourant) {
		case Start:
			etat = "Init";
			break;

		case Run:
			etat = "Run";
			break;

		case Pause:
			etat = "Pause";
			break;

		case GameOver:
			etat = "GameOver";
			break;

		case Victoire:
			etat = " Victoire";
			break;
			
		default:
			etat = "il y a un probleme";
			break;
		}

		return etat;

	}

}
