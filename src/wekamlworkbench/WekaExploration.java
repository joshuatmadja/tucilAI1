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

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
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
        System.out.println(eval.toSummaryString("\nRingkasan\n=========", true));
        System.out.println(eval.toMatrixString("\nMatriks\n======="));
    }
    /**
     * @param args the command line arguments
     * @throws java.io.FileNotFoundException
     */
     public static void main(String[] args) throws Exception {
        WekaExploration wk = new WekaExploration();
        Discretize filter = new Discretize();
        Scanner in = new Scanner(System.in);
        
        wk.readInstances();
        Instances irisDiscretized = wk.getDiscretized();
        
        //wk.visualize(irisDiscretized);
        wk.tenCrossValidate(irisDiscretized);
        wk.fullTraining(irisDiscretized);
    }   
}