package wekamlworkbench;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;
import java.util.Scanner;
import weka.classifiers.Classifier;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;

public class WekaExploration{
    Instances iris;
    Evaluation eval;
    Scanner in = new Scanner(System.in);
    J48 pohon = new J48();
    
    public void readInstances() throws FileNotFoundException, IOException, Exception{
        BufferedReader b = new BufferedReader(new FileReader ("C:/Program Files/Weka-3-8/data/iris.arff"));
        iris = new Instances (b);
        iris.setClassIndex(iris.numAttributes()-1);
        
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
        pohon.buildClassifier(i);
        eval = new Evaluation(i);
        eval.crossValidateModel(pohon, i, 10, new Random(1));
        System.out.println("##10 FOLDS CROSS VALIDATION##");
        System.out.println(eval.toSummaryString("\nRingkasan\n=========", true));
        System.out.println(eval.toMatrixString("\nMatriks\n======="));
    }
    
    
    public void fullTraining(Instances i) throws Exception{
        pohon.buildClassifier(i);
        eval = new Evaluation(i);
        eval.evaluateModel(pohon, i);
        System.out.println("##FULL TRAINING##");
        System.out.println(eval.toClassDetailsString("\nRincian\n======="));
        System.out.println(eval.toSummaryString("\nRingkasan\n=========", true));
        System.out.println(eval.toMatrixString("\nMatriks\n======="));
    }
    
    public void saveModel(Instances i, String f) throws Exception{
        eval = new Evaluation(i);
        pohon = new J48();
        pohon.buildClassifier(i);
        SerializationHelper.write(f+".model", pohon);
    }
    
    public Classifier readModel(Instances current, String f) throws FileNotFoundException, Exception {
        Classifier cls = null;
        try{  
            cls = (Classifier) SerializationHelper.read(f+".model");
            System.out.println(f+".model berhasil dibaca\n");
            System.out.println("\nModel yang terbaca\n==================\n"+cls.toString());
        }
        catch (FileNotFoundException e){
            System.out.println("Berkas "+f+".model tidak ditemukan\n");
        }
        return cls;
    }
    
    public void addInstances(Instances inst){
        Instance newInst;
        double newinstance[] = new double[inst.numAttributes()];
        String[] attrname = {"sepallength","sepalwidth","petallength","petalwidth"};
        for(int i = 0; i < inst.numAttributes()-1; i++){
            System.out.print(attrname[i]+": ");
            double value = in.nextDouble();
            newinstance[i]=value;
        }
        newInst = new DenseInstance(1.0, newinstance);
        inst.add(newInst);
    }
    
    public void askToSave(Instances i) throws Exception{
        System.out.print("Tulis nama berkasnya (*.model): ");
        String filename;
        filename = in.next();
        saveModel(i, filename);
        System.out.println(filename+".model sudah tersimpan");
        System.out.println("\n");
    }
    
    public void classifyInstance(Instances i) throws Exception{
        double classIndex = pohon.classifyInstance(i.lastInstance());
        int lastIdx = i.numAttributes()-1;
        String classN = i.attribute(lastIdx).value( (int) classIndex);
        i.instance(i.numInstances()-1).setClassValue(classN);
        System.out.println(i.lastInstance()+"\n");
    }
    
    public static void main(String[] args) throws Exception {
        WekaExploration wk = new WekaExploration();
        boolean built=false;
        int end = 0;
        String save;
        Scanner in = new Scanner(System.in);
        wk.readInstances();
        System.out.println("Pada program ini, dataset yang dibaca ialah \'iris.arff\' dan menggunakan filter Supervised Discretized.\n");
        
        while (end==0) {
        Instances irisDiscretized = wk.getDiscretized();
        System.out.println("Apa yang ingin Anda lakukan?");
        System.out.println("1. Tampilkan hasil Full Training");
        System.out.println("2. Tampilkan hasil 10 Fold Cross Validation");
        System.out.println("3. Baca model");
        System.out.println("4. Menyisipkan sebuah instance ke dalam dataset");
        System.out.println("5. Keluar");
        System.out.print("Pilihan Anda: ");
        int c;
        c = in.nextInt();
        System.out.println();
        switch (c) {
            case 1:
                wk.fullTraining(irisDiscretized);
                built=true;
                System.out.print("Apakah anda ingin menyimpan model ini? (y/n): ");
                save = in.next();
                if(save.equals("y") || save.equals("Y"))  wk.askToSave(irisDiscretized);
                System.out.println();
                break;
            case 2:
                wk.tenCrossValidate(irisDiscretized);
                built=true;
                System.out.print("Apakah anda ingin menyimpan model ini? (y/n): ");
                save = in.next();
                if(save.equals("y") || save.equals("Y"))  wk.askToSave(irisDiscretized);
                System.out.println();
                break;
            case 3 :
                System.out.print("Masukkan nama model yang ingin dibaca (*.model): ");
                String filenames;
                filenames = in.next();
                wk.pohon = (J48) wk.readModel(irisDiscretized, filenames);
                break;
            case 4:
                if(!built){
                    wk.fullTraining(wk.iris);
                    built=true;
                }
                System.out.println("Masukkan instance berikut\n===========");
                wk.addInstances(wk.iris);
                System.out.println("\nInstance berhasil dimasukkan\n");
                System.out.print("Klasifikasi instance baru: ");
                wk.classifyInstance(wk.iris);
                break;
            case 5 :
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