/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wekamlworkbench;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;
import java.util.Scanner;
import weka.classifiers.Classifier;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;

/**
 *
 * @author user
 */
public class WekaExploration{
    Instances iris;
    Evaluation eval;
    
    public void readInstances() throws FileNotFoundException, IOException, Exception{
        try (BufferedReader b = new BufferedReader(new FileReader ("C:/Program Files/Weka-3-8/data/iris.arff"))) {
            iris = new Instances (b);
            iris.setClassIndex(iris.numAttributes()-1);
        }
    }
   
    public Instances getDiscretized() throws Exception{
        Discretize d = new Discretize();
        Instances i;
        
        d.setInputFormat(iris);
        i = Filter.useFilter(iris, d);
        i.setClassIndex(i.numAttributes()-1);
        return i;
    }
    
    public void tenCrossValidate(Instances i) throws Exception{
        eval = new Evaluation(i);
        NaiveBayes nB = new NaiveBayes();
        nB.buildClassifier(i);
        eval.crossValidateModel(nB, i, 10, new Random(1));
        System.out.println("##10 FOLDS CROSS VALIDATION##");
        System.out.println(eval.toSummaryString("\nRingkasan\n=========", true));
        System.out.println(eval.toMatrixString("\nMatriks\n======="));
        //System.out.println(eval.fMeasure(1)+" "+eval.precision(1)+" "+eval.recall(1));
    }
    
    
    public void fullTraining(Instances i) throws Exception{
        eval = new Evaluation(i);
        NaiveBayes nB = new NaiveBayes();
        nB.buildClassifier(i);
        eval.evaluateModel(nB, i);
        System.out.println("##FULL TRAINING##");
        System.out.println(eval.toClassDetailsString("\nRincian\n======="));
        System.out.println(eval.toSummaryString("\nRingkasan\n=========", true));
        System.out.println(eval.toMatrixString("\nMatriks\n======="));
    }
    
    public void saveModel(Instances i, String f) throws Exception{
        eval = new Evaluation(i);
        NaiveBayes nB = new NaiveBayes();
        nB.buildClassifier(i);
        SerializationHelper.write(f+".model", nB);
    }
    
    public Classifier readModel(Instances current, String f) throws FileNotFoundException, Exception {
        Classifier cls = (Classifier) SerializationHelper.read(f+".model");     
        return cls;
    }
    
     public static void main(String[] args) throws Exception {
        WekaExploration wk = new WekaExploration();
        Discretize filter = new Discretize();
        Scanner in = new Scanner(System.in);
        int run = 0;
        int end = 0;
        
        System.out.println("Pada program ini, dataset yang dibaca ialah \'iris.arff\' dan menggunakan filter Supervised Discretized.\n");
        
        while (end==0) {
        wk.readInstances();
        Instances irisDiscretized = wk.getDiscretized();
        System.out.println("Apa yang ingin Anda lakukan?");
        System.out.println("1. Tampilkan hasil Full Training");
        System.out.println("2. Tampilkan hasil 10 Fold Cross Validation");
        System.out.println("3. Simpan model");
        System.out.println("4. Baca model");
        System.out.println("5. Keluar");
        System.out.print("Pilihan Anda: ");
        int c;
        c = in.nextInt();
        System.out.println();
        //wk.visualize(irisDiscretized);
        switch (c) {
            case 1:
                wk.fullTraining(irisDiscretized);
                run = 1;
                break;
            case 2:
                wk.tenCrossValidate(irisDiscretized);
                run = 1;
                break;
            case 3:
                if (run==0) {
                    System.out.println("Anda belum menampilkan dataset");
                    System.out.println();
                }
                System.out.print("Tulis nama berkasnya (*.model): ");
                String filename;
                filename = in.next();
                wk.saveModel(irisDiscretized, filename);
                System.out.println(filename+".model sudah tersimpan");
                System.out.println();
                break;
            case 4 :
                System.out.print("Masukkan nama model yang ingin dibaca : ");
                String filenames;
                filenames = in.next();
                wk.readModel(irisDiscretized, filenames);
                System.out.println(filenames+".model berhasil dibaca");
                System.out.println();
                break;
            case 5 :
                end = 1;
                System.out.println("Terima kasih");
                break;
            default:
                System.out.println("Pilihan tidak tersedia");
                break;
        }
        }
        
    }   
}