package motor;

public class SimpleGame extends Game {

	public SimpleGame(int maximum_laps) {
		super(maximum_laps);
	}

	public SimpleGame(int maximum_laps, long speed) {
		super(maximum_laps, speed);
	}


	@Override
	public void takeTurn() {}

	@Override
	public void gameOver() {}

	@Override
	public void reinitializeGameStat() {
		// TODO Auto-generated method stub
		
	}

}
