package gui;

//import data.ReadWriteFile;
import javax.swing.*;
import java.awt.*;
import java.util.ArrayList;
//import java.util.Vector;
import data.Vector;
import neural.NeuralNet;
import neural.NeuralNetTest;

public class MainGui extends JFrame {

    private final int RESOLUTION = 28;

    private int resultat;
    private JPanel mainPanel;
    private DrawingPanel drawingPanel;

    private JButton clearButton;
    private JButton transformButton;
    private JTextArea outputTextArea;
    private ArrayList<Double> userInputs = new ArrayList<>();
    private NeuralNetTest neuralNetTest;
    
    public static void main(String[] args) {
    	MainGui mainGui = new MainGui();
    }

    public MainGui() {
        super("Drawing digits using neural networks");

        setMainPanel();
        setLeftSide();
        setCenterArea();
        setOutputPanel();
        
        setOnClicks();

        setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        setVisible(true);
        setSize(new Dimension(800, 500));
        setLocationRelativeTo(null);
        setResizable(false);
        neuralNetTest = new NeuralNetTest();
        neuralNetTest.train();
    }

    private void setMainPanel() {
        mainPanel = new JPanel();
        mainPanel.setBackground(Color.darkGray);
        setContentPane(mainPanel);
    }

    private void setLeftSide() {
        JPanel panel = new JPanel();
        panel.setBackground(Color.DARK_GRAY);
        panel.setPreferredSize(new Dimension(410, 440));

        drawingPanel = new DrawingPanel(400, 400, RESOLUTION);

        panel.add(drawingPanel);

        mainPanel.add(panel);
    }

    private void setCenterArea() {
        JPanel centerPanel = new JPanel();
        centerPanel.setLayout(new GridBagLayout());
        centerPanel.setPreferredSize(new Dimension(100, 200));
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.gridwidth = GridBagConstraints.REMAINDER;
        gbc.anchor = GridBagConstraints.CENTER;

        centerPanel.add(Box.createVerticalStrut(50));

        centerPanel.add(Box.createVerticalStrut(50));

        transformButton = new JButton(">>");
        centerPanel.add(transformButton, gbc);

        centerPanel.add(Box.createVerticalStrut(50));

        clearButton = new JButton("Clear");
        clearButton.setAlignmentX(Component.CENTER_ALIGNMENT);
        centerPanel.add(clearButton, gbc);

        centerPanel.add(Box.createVerticalStrut(50));

        mainPanel.add(centerPanel);
    }


    private void setOutputPanel() {
        JPanel outputPanel = new JPanel();
        outputPanel.setPreferredSize(new Dimension(200, 430));

        outputTextArea = new JTextArea();
        outputTextArea.setPreferredSize(new Dimension(200, 430));
        outputPanel.add(outputTextArea);

        mainPanel.add(outputPanel);
    }

    private void setOnClicks() {
        clearButton.addActionListener(e -> drawingPanel.clear());

        transformButton.addActionListener(e -> {
//        	for(Integer i : drawingPanel.getPixels()) {
//        		this.userInputs.add(i.doubleValue()); //this.////?
//        	}
        	userInputs.clear();
        	for (int i =0; i<drawingPanel.getPixels().size(); i++) {
        		ArrayList<Integer> pixelList = drawingPanel.getPixels();
        		Integer pixel = pixelList.get(i);
        		Double dblPixel = pixel.doubleValue();
        		this.userInputs.add(dblPixel);
        	}
        	
        	neuralNetTest.testDigits(userInputs);
        	this.resultat = neuralNetTest.getRes();
            updateTextArea();
        });

    }

    private void passInputs(ArrayList<Double> userInputs) {
    	
    }
    private void updateTextArea() {
        StringBuilder sb = new StringBuilder();
        int value;
        value = this.resultat;

        sb.append("\t " + value);
        sb.append("\n");
        outputTextArea.setText(sb.toString());
        }
        
	}
