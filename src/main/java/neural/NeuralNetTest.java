package neural;

import learn.Backpropagation;
import gui.MainGui;
import learn.Training;
import learn.Training.ActivationFncENUM;
import learn.Training.TrainingTypesENUM;
import data.LoadMNIST;
import data.Vector;

import java.util.ArrayList;

import org.apache.log4j.Logger;

public class NeuralNetTest {
	static Logger log = Logger.getLogger(NeuralNetTest.class);
	int res;
	private NeuralNet testNet = new NeuralNet();
	
	
	public void train() {
		log.debug("BEGIN");
		
		NeuralNetTest thing = new NeuralNetTest();
		LoadMNIST mnistLoader = new LoadMNIST();
		
		testNet = testNet.initNet(784, 1, 10, 10);
		log.debug("---------BACKPROPAGATION INIT NET---------");
		
		testNet.printNet(testNet);
		
		NeuralNet trainedNet = new NeuralNet(); 
		
		Vector[] vectTrain = mnistLoader.importData("train-images-idx3-ubyte.gz");
		double[][] masTrainSet = new double[vectTrain.length][];
		thing.InitArrayBias(vectTrain, masTrainSet);
		
		Vector[] vectRealOut = mnistLoader.importData("train-labels-idx1-ubyte.gz");
		double[][] masRealOutput = new double[vectRealOut.length][];
		thing.InitArray(vectRealOut, masRealOutput);
		
		testNet.setTrainSet(masTrainSet);
		testNet.setRealMatrixOutputSet(masRealOutput);
		
		testNet.setMaxEpochs(1);
		testNet.setTargetError(0.001); 
		testNet.setLearningRate(0.05);
		testNet.setTrainType(TrainingTypesENUM.BACKPROPAGATION); 
		testNet.setActivationFnc(ActivationFncENUM.SIGLOG);
		testNet.setActivationFncOutputLayer(ActivationFncENUM.SIGLOG);
		
		trainedNet = testNet.trainNet(testNet);
		
		log.debug("---------BACKPROPAGATION INIT NET---------");
		log.debug("/n");
		

		testNet.printNet(trainedNet);
		log.debug("END OF TRAINING");

		Vector[] vectTestIm = mnistLoader.importData("t10k-images-idx3-ubyte.gz");
		double[][] masTestIm = new double[vectTestIm.length][];
		thing.InitArrayBias(vectTestIm, masTestIm);
				
		Vector[] vectTestLb = mnistLoader.importData("t10k-labels-idx1-ubyte.gz");
		double[][] masTestLb = new double[vectTestLb.length][];
		thing.InitArray(vectTestLb, masTestLb);
		
		testNet.setTrainSet(masTestIm);
		testNet.setRealMatrixOutputSet(masTestLb);
		
		Boolean bool = true;
		Backpropagation b = new Backpropagation();
		for(int j=0; j< vectTestLb.length; j++)
		b.forward(testNet, j, bool);
		bool = false;
	}
	
	public void testDigits(ArrayList<Double> example) {		
		LoadMNIST mnistLoader = new LoadMNIST();
		NeuralNetTest thing = new NeuralNetTest();
		
		double[][] masTestIm = new double[786][786];
		for(int i=0; i< example.size(); i++) {
			masTestIm[0][i] = example.get(i);
		}
		testNet.setTrainSet(masTestIm);
		
		Backpropagation b = new Backpropagation();
		Boolean y = false;
		b.forward(testNet, 0, y);
		this.res = b.getResult();
	}
	
	private void InitArrayBias(Vector[] vect, double[][] mas) {
		for(int i=0; i< vect.length; i++){
			mas[i] = new double[vect[i].getN()+1];
		 	mas[i][0]=1.0; 
			for(int j=0; j< (vect[i].getN());j++) {
				mas[i][j+1]=vect[i].get(j);
			}
		}
	} 
	
	private void InitArray(Vector[] vect, double[][] mas) {
		for(int i=0; i< vect.length; i++){
			mas[i] = new double[vect[i].getN()];
			for(int j=0; j< (vect[i].getN());j++) {
				mas[i][j]=vect[i].get(j);
			}
		}
	} 
    public int getRes() {
       return this.res;
    }
        
    public NeuralNet getNeuralNet() {
       return testNet;
    }
        
}


