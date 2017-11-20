package data;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
//import java.util.Vector;
import java.util.zip.GZIPInputStream;
import org.apache.log4j.Logger;

import neural.NeuralNetTest;

public class LoadMNIST { 

    public static Vector[] importData(String fileName) { 
    	Logger log = Logger.getLogger(LoadMNIST.class);
        try {
        	log.debug("\n---Importing MNIST data---\nfile: " + fileName);
            //System.out.println("\n---Importing MNIST data---\nfile: " + fileName);
            String fullName = LoadMNIST.class.getClassLoader().getResource(fileName).getFile();
            File zipFile = new File(fullName);
            GZIPInputStream gzip = new GZIPInputStream(new FileInputStream(zipFile));
            byte[] magic = new byte[4];
            gzip.read(magic);
            int magicNum = bytesToInt(magic);
            log.debug("magic num: " + magicNum);
            switch (magicNum) {
                case 2049:
                    return importLabelFile(gzip);
                case 2051:
                    return importImageFile(gzip);
                default:
                    log.error("This is not a valid file. magic num: " + magicNum);
                    break;
            }
        } catch (IOException ex) {
            log.error("Error while reading file:\n" + ex);
        }
        return new Vector[0];
    }

     private static int bytesToInt(byte[] bytes) { 
        return ((bytes[0] & 0xFF) << 24 | (bytes[1] & 0xFF) << 16 | (bytes[2] & 0xFF) << 8 | (bytes[3] & 0xFF));
    }

     private static Vector[] importLabelFile(GZIPInputStream gzip) throws IOException { 
    	Logger log = Logger.getLogger(LoadMNIST.class);
        byte[] itemCountBytes = new byte[4];
        gzip.read(itemCountBytes);
        int itemCount = bytesToInt(itemCountBytes);
        log.debug("item count: " + itemCount);
        Vector[] data = new Vector[itemCount];
        for (int i = 0; i < itemCount; i++) {
            double[] vec = new double[10];
            vec[gzip.read()] = 1.0;
            data[i] = new Vector(vec);
        }
        log.debug("finished");
        return data;
    }

    private static Vector[] importImageFile(GZIPInputStream gzip) throws IOException {
    	Logger log = Logger.getLogger(LoadMNIST.class);
        byte[] infoBytes = new byte[4];
        gzip.read(infoBytes);
        int itemCount = bytesToInt(infoBytes);
        gzip.read(infoBytes);
        int rowCount = bytesToInt(infoBytes);
        gzip.read(infoBytes);
        int colCount = bytesToInt(infoBytes);
        log.debug("item count: " + itemCount);
        log.debug("row count: " + rowCount);
        log.debug("col count: " + colCount);
        Vector[] data = new Vector[itemCount];
        int pixelCount = rowCount * colCount;
        for (int i = 0; i < itemCount; i++) {
            double[] vec = new double[pixelCount];
            for (int j = 0; j < pixelCount; j++) {
                vec[j] = gzip.read() / 255.0;
            }
            data[i] = new Vector(vec);
        }
        log.debug("finished");
        return data;
    } 
}