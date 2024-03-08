package view;

import java.awt.Dimension;
import java.awt.GraphicsEnvironment;
import java.awt.GridLayout;
import java.awt.Point;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;

import javax.swing.Icon;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JPanel;
import javax.swing.JSlider;
import javax.swing.SwingConstants;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import agent.AgentAction;

import motor.Maze;
import motor.PacmanGame;
import motor.State;

import observator_pattern.Observer;

import controller.GameController;

public class View implements Observer {

	private static View uniqueInstance;
	
	private GameController _controller;
	public PacmanGame _motor;

	private PanelPacmanGame _panelPacmanGame;
	private JFrame window;
	private JFrame affichage;
	private JPanel panelPrincipal;
	private JPanel panelHaut;
	private JPanel panelBas;
	private JPanel panelBasGauche;
	private JPanel panelGame;
	private JPanel panelGameOver;
	private GridLayout layoutPrincipal;
	private GridLayout layoutHaut;
	private GridLayout layoutBas;
	private GridLayout layoutBasGauche;
	private JButton btnRestart;
	public JButton btnPause;
	public JButton btnRun;
	private JButton btnStep;
	private JSlider slider;
	private JLabel text;
	private JLabel sliderLabel;
	private JLabel gameText;
	private JLabel game_over;

	private JMenuBar menuBar;
	private JMenu menu;
	private JMenuItem choisirLabyrinthe;
	private JFileChooser choixLabyrinthe;

	static final int slider_min = 0;
	static final int slider_max = 10;
	static final int slider_init = 1;

	private String chemin_maze;

	private View(GameController controller, PacmanGame motor, boolean modeControl) {

		
		_panelPacmanGame = new PanelPacmanGame(motor.getMaze());
		
		
		_controller = controller;
		_motor = motor;
		_motor.addObserver(this);


		initBtn();
		initText();
		initGameText();
		initSlider();

		
		initPanelBasGauche();
		initPanelBas();
		initPanelHaut();
		initPanelPrincipal();
		
		
		initPanelGame();

		initMenu();

		initWindow();
		initAffichage();

		initListener();

		//affichage.addKeyListener(this);
		window.setVisible(true);
		affichage.setVisible(true);

		

	}

	public static View getInstance(GameController controller, PacmanGame motor, boolean modeControl) {
		
		if(uniqueInstance == null) {
			uniqueInstance = new View(controller, motor, modeControl);
			
		}
		
		return uniqueInstance;
		
		
	}
	

	
	@Override
	public void update() {
		setTurn(_motor.get_counter_nb_laps(), _motor.getScore());
		setGameText("Tour " + _motor.get_counter_nb_laps() + " etat courant "
				+ _motor.getEtatString());

		if (_motor.getEtatCourant() != State.GameOver) {
			if (_motor.getEtatCourant() != State.Victoire) {
				// System.out.println("dans update view");
				// on actualise les données du panel avec celle du pacman game
				_panelPacmanGame.setPacmans_pos(_motor.getPostionPacman());
				_panelPacmanGame.setGhosts_pos(_motor.getPostionFantom());
				_panelPacmanGame.setGhostsScarred(_motor.isGhostsScarred());
				
				_panelPacmanGame.setMaze(_motor.getMaze());
				_panelPacmanGame.repaint();

				if (affichage != null) {
					affichage.getContentPane().add(_panelPacmanGame);
					affichage.setVisible(true);
				}
			} else {

				
				btnRun.setEnabled(false);
				btnRestart.setEnabled(true);
				btnPause.setEnabled(false);
				btnStep.setEnabled(false);

				
			}
		} else {
			btnRun.setEnabled(false);
			btnRestart.setEnabled(true);
			btnPause.setEnabled(false);
			btnStep.setEnabled(false);
		}

	}

	public void setTurn(int turn, int point) {
		text.setText("Number of laps : " + turn + "     Points : " + point);
	}

	public void setGameText(String text) {
		gameText.setText(text);
	}

	/**//**//**//**//**//**//**//**//**//**//**//**/// Listeners

	private void initListener() {
		btnRestart.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent arg0) {
				btnRun.setEnabled(true);
				btnRestart.setEnabled(false);
				btnPause.setEnabled(false);
				btnStep.setEnabled(true);

//				setMaze(chemin_maze);
				_controller.restart();
			}
		});
		btnRun.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent arg0) {
				btnRun.setEnabled(false);
				btnRestart.setEnabled(false);
				btnPause.setEnabled(true);
				btnStep.setEnabled(false);

				_controller.start();
			}
		});

		btnStep.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent arg0) {
				btnRun.setEnabled(true);
				btnRestart.setEnabled(true);
				btnPause.setEnabled(false);
				btnStep.setEnabled(true);

				_controller.step();
			}
		});
		btnPause.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent arg0) {
				btnRun.setEnabled(true);
				btnRestart.setEnabled(true);
				btnPause.setEnabled(false);
				btnStep.setEnabled(true);

				_controller.pause();
			}
		});

		slider.addChangeListener(new ChangeListener() {
			@Override
			public void stateChanged(ChangeEvent arg0) {
				_controller.speed(slider.getValue()+1);
			}
		});

		// Menu Item Functions
		choisirLabyrinthe.addActionListener(new ActionListener() {
			@SuppressWarnings("deprecation")
			public void actionPerformed(ActionEvent arg0) {
				int choix;

				choix = choixLabyrinthe.showOpenDialog(null);

				if (choix == JFileChooser.APPROVE_OPTION) {
					chemin_maze = choixLabyrinthe.getSelectedFile().getPath();

					affichage.getContentPane().removeAll();

//					setMaze(chemin_maze);
					affichage.getContentPane().add(_panelPacmanGame);
					affichage.resize(
							_panelPacmanGame.getMaze().getSizeX() * 30,
							_panelPacmanGame.getMaze().getSizeY() * 30);
					affichage.setVisible(true);
				}
			}
		});
	}

	/**//**//**//**//**//**//**//**//**//**//**//**/// inits

	public void initMenu() {
		menuBar = new JMenuBar();
		menu = new JMenu("Menu");
		choisirLabyrinthe = new JMenuItem("Choisir un nouveau labyrinthe");
		choixLabyrinthe = new JFileChooser("src/layout");

		menu.add(choisirLabyrinthe);
		menuBar.add(menu);
	}

	public void initWindow() {
		window = new JFrame();
		window.setTitle("Commande");
		window.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		window.setSize(new Dimension(700, 300));

		Dimension windowSize = window.getSize();
		GraphicsEnvironment ge = GraphicsEnvironment
				.getLocalGraphicsEnvironment();
		Point centerPoint = ge.getCenterPoint();
		int dx = centerPoint.x  - (int)(windowSize.width*1.3) ;
		int dy = centerPoint.y - windowSize.height / 2;
		window.setLocation(dx, dy);

		window.add(panelPrincipal);
	}

	public void initAffichage() {

//		chemin_maze = "src/layout/originalClassic.lay";
//		setMaze(chemin_maze);

		affichage = new JFrame();
		affichage.setTitle("affichage");
		affichage.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		affichage.setSize(_panelPacmanGame.getMaze().getSizeX() * 30,
				_panelPacmanGame.getMaze().getSizeY() * 60);

		Dimension windowSize = affichage.getSize();
		GraphicsEnvironment ge = GraphicsEnvironment
				.getLocalGraphicsEnvironment();
		Point centerPoint = ge.getCenterPoint();
		int dx = centerPoint.x ;
		int dy = centerPoint.y - windowSize.height / 2;
		affichage.setLocation(dx, dy);

		affichage.add(_panelPacmanGame);
		affichage.setJMenuBar(menuBar);

		affichage.setVisible(true);
		
	}

	public void initPanelPrincipal() {
		panelPrincipal = new JPanel();
		layoutPrincipal = new GridLayout(2, 1);
		panelPrincipal.setLayout(layoutPrincipal);

		panelPrincipal.add(panelHaut);
		panelPrincipal.add(panelBas);

	}

	public void initPanelHaut() {
		panelHaut = new JPanel();
		layoutHaut = new GridLayout(1, 4);
		panelHaut.setLayout(layoutHaut);

		panelHaut.add(btnRestart);
		panelHaut.add(btnRun);
		panelHaut.add(btnStep);
		panelHaut.add(btnPause);
	}

	public void initPanelBas() {
		panelBas = new JPanel();
		layoutBas = new GridLayout(1, 2);
		panelBas.setLayout(layoutBas);

		panelBas.add(panelBasGauche);
		panelBas.add(text);
	}

	public void initPanelBasGauche() {
		panelBasGauche = new JPanel();
		layoutBasGauche = new GridLayout(2, 1);
		panelBasGauche.setLayout(layoutBasGauche);

		panelBasGauche.add(sliderLabel);
		panelBasGauche.add(slider);
	}

	public void initPanelGame() {
		panelGame = new JPanel();

		panelGame.add(gameText);
	}

	public void initBtn() {
		Icon icon_restart = new ImageIcon("src/icones/icon_restart.png");
		Icon icon_run = new ImageIcon("src/icones/icon_run.png");
		Icon icon_step = new ImageIcon("src/icones/icon_step.png");
		
		Icon icon_pause = new ImageIcon("src/icones/icon_pause.png");

		btnRestart = new JButton(icon_restart);
		btnRun = new JButton(icon_run);
		btnStep = new JButton(icon_step);
		btnPause = new JButton(icon_pause);

		btnRestart.setEnabled(false);
		btnStep.setEnabled(true);
		btnPause.setEnabled(false);
	}

	public void initSlider() {
		sliderLabel = new JLabel("Number of turn per second", JLabel.CENTER);
		slider = new JSlider(JSlider.HORIZONTAL, slider_min, slider_max,
				slider_init);

		slider.setMajorTickSpacing(1);
		slider.setMinorTickSpacing(1);
		slider.setPaintTicks(true);
		slider.setPaintLabels(true);
	}

	public void initText() {
		text = new JLabel("", SwingConstants.CENTER);
		setTurn(0, 0);
	}

	public void initGameText() {
		gameText = new JLabel("", SwingConstants.CENTER);
		setGameText("Situation initiale");
	}

//	public void setMaze(String chemin) {
//		try {
//			Maze newMaze = new Maze(chemin);
//			
//			_motor.setMaze(newMaze);
//		} catch (Exception e) {
//			// TODO Auto-generated catch block
//			e.printStackTrace();
//		}
//	}

	/*
	 * 
	 * Gestion du keylistener
	 */

//	@Override
//	public void keyPressed(KeyEvent e) {
//
//		switch (e.getKeyCode()) {
//		case KeyEvent.VK_LEFT:
//			_motor.set_next_move_J1(AgentAction.WEST);
//			break;
//		case KeyEvent.VK_RIGHT:
//			_motor.set_next_move_J1(AgentAction.EAST);
//			break;
//		case KeyEvent.VK_UP:
//			_motor.set_next_move_J1(AgentAction.NORTH);
//			break;
//		case KeyEvent.VK_DOWN:
//			_motor.set_next_move_J1(AgentAction.SOUTH);
//			break;
//		default:
//			break;
//		}
//
//	}

//	@Override
//	public void keyReleased(KeyEvent e) {
//		// _motor.set_next_move_J1(AgentAction.STOP);
//
//	}
//
//	@Override
//	public void keyTyped(KeyEvent e) {
//		// TODO Auto-generated method stub
//
//	}

}
