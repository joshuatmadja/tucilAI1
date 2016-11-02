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
    
    public void readModel(Instances current, String f) throws FileNotFoundException, Exception {
        try{  
            Classifier cls = (Classifier) SerializationHelper.read(f+".model");
            System.out.println(f+".model berhasil dibaca\n");
            System.out.println("\nModel yang terbaca\n==================\n"+cls.toString());
        }
        catch (FileNotFoundException e){
            System.out.println("File "+f+".model tidak ditemukan\n");
        }
    }
    
    public void askToSave(Instances i) throws Exception{
        Scanner in = new Scanner(System.in);
        System.out.print("Tulis nama berkasnya (*.model): ");
        String filename;
        filename = in.next();
        saveModel(i, filename);
        System.out.println(filename+".model sudah tersimpan");
        System.out.println();
    }
    
    public static void main(String[] args) throws Exception {
        WekaExploration wk = new WekaExploration();
        Discretize filter = new Discretize();
        Scanner in = new Scanner(System.in);
        int end = 0;
        String save;
        
        System.out.println("Pada program ini, dataset yang dibaca ialah \'iris.arff\' dan menggunakan filter Supervised Discretized.\n");
        
        while (end==0) {
        wk.readInstances();
        Instances irisDiscretized = wk.getDiscretized();
        System.out.println("Apa yang ingin Anda lakukan?");
        System.out.println("1. Tampilkan hasil Full Training");
        System.out.println("2. Tampilkan hasil 10 Fold Cross Validation");
        System.out.println("3. Baca model");
        System.out.println("4. Keluar");
        System.out.print("Pilihan Anda: ");
        int c;
        c = in.nextInt();
        System.out.println();
        //wk.visualize(irisDiscretized);
        switch (c) {
            case 1:
                wk.fullTraining(irisDiscretized);
                System.out.print("Apakah anda ingin menyimpan model ini? (y/n): ");
                save = in.next();
                if(save.equals("y") || save.equals("Y"))  wk.askToSave(irisDiscretized);
                break;
            case 2:
                wk.tenCrossValidate(irisDiscretized);
                System.out.print("Apakah anda ingin menyimpan model ini? (y/n): ");
                save = in.next();
                if(save.equals("y") || save.equals("Y"))  wk.askToSave(irisDiscretized);
                break;
            case 3 :
                System.out.print("Masukkan nama model yang ingin dibaca (*.model): ");
                String filenames;
                filenames = in.next();
                wk.readModel(irisDiscretized, filenames);
                break;
            case 4 :
                end = 1;
                System.out.println("Terima kasih");
                break;
            default:
                System.out.println("Pilihan tidak tersedia. Ulangi lagi.\n");
                break;
        }
        }
        
    }   
}