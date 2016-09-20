
package meka.classifiers.multilabel;

import java.text.DecimalFormat;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.*;
import java.io.*;

import meka.classifiers.multilabel.BR;
import meka.classifiers.multilabel.Evaluation;
import weka.classifiers.functions.Logistic;
import weka.classifiers.trees.J48;
import weka.classifiers.Classifier;
import weka.core.SerializationHelper;
import meka.core.MLEvalUtils;
import meka.core.MLUtils;
import meka.core.ThresholdUtils;
import meka.core.Result;
import weka.core.Utils;
import weka.core.Instances;
import weka.core.Instance;
import weka.core.converters.ConverterUtils.DataSource;
import rbms.Mat;

/**
 * This example shows how you to save a learned model and load a stored model.
 */

public class MLThresholdChoiceMethods
{
    /**
     * 
     * @param args command-line arguments -train, -test and -model, e.g. -train Flags-train.arff -test Flags-test.arff -model model.dat
     */
    public static void main(String[] args)
    {
        try
        {
        	String datasetList[] ={"Corel5k","CAL500","Bibtex","LLOG","Enron","Medical","Genbase","SLASHDOT","Birds","Yeast","Flags","Emotions","Scene"};
      	    String path="/Users/ra12404/Desktop/meka-1.7.5/data/Thresholds/";
      	    String Mydataset;      	    
      	    DecimalFormat df = new DecimalFormat("#.###");    

    	    BufferedWriter writer = new BufferedWriter(new FileWriter("/Users/ra12404/Desktop/meka-1.7.5/data/Thresholds/results.csv"));
            PrintWriter SDUout= new PrintWriter(new BufferedWriter(new FileWriter("/Users/ra12404/Desktop/meka-1.7.5/data/Thresholds/SDU_results.csv")));

	 		writer.write("Dataset");
	 		writer.write(',');
	 		writer.write("Fixed");
	 		writer.write(',');
	 		writer.write("PCut");
	 		writer.write(',');
	 		writer.write("SD");
	 		writer.write(',');
	 		writer.write("SCut");
	 		writer.write(',');
	 		writer.write("L PCut");
	 		writer.write(',');
	 		writer.write("L SD");
	 		writer.write(',');
	 		writer.write("L SCut");
	 		writer.write(',');
	 		writer.write("L SDU");
	 		writer.write(',');
	 		writer.write("L RCut");
	 		writer.write(',');
	 		writer.write("L MCut");
	 		writer.write(',');
	 		writer.write('\n');	
      	    
	 		SDUout.write("Dataset");
	 		SDUout.write(',');
	 		SDUout.write("Selected label");
	 		SDUout.write(',');
	 		SDUout.write("avg_loss");
	 		SDUout.write(',');
	 		SDUout.write("avg_BR");
	 		SDUout.write(',');
	 		SDUout.write("avg_MAE");
	 		SDUout.write(',');
	 		SDUout.write("1/q[avg_BR+(q-1)avg_MAE]");
	 		SDUout.write(',');
	 		SDUout.write('\n');	
	 		
      	  for (int dd=0; dd<datasetList.length; dd++)
	 		{
      		Mydataset=datasetList[dd];
      		writer.write(Mydataset);
	 		writer.write(',');
	 		
        	//Train the model and test it
        	String trainingDataFilename,testingDataFilename;
      	    DataSource sourcetrain,sourcetest;
      	    Instances D_train,D_test; 
      	    Result r = new Result();      	   
      	  
      	    /* Training data */
      	    trainingDataFilename = path+Mydataset+"-train.arff";
	    	sourcetrain = new DataSource(trainingDataFilename);
	    	D_train =  sourcetrain.getDataSet(); 
	    	MLUtils.prepareData(D_train);	
	    	
	    	/* Deployment data */
	    	testingDataFilename = path+Mydataset+"-test.arff";				
			sourcetest = new DataSource(testingDataFilename);
			D_test =  sourcetest.getDataSet(); 
			MLUtils.prepareData(D_test); 
			
			PrintWriter out= new PrintWriter(new BufferedWriter(new FileWriter("/Users/ra12404/Desktop/meka-1.7.5/data/Thresholds/"+Mydataset+"/J48/"+"compare.txt")));
            Logistic baseClassifier_LR=new Logistic(); //base classifier
            baseClassifier_LR.setRidge(100);
            
            J48 baseClassifier_J48=new J48(); //base classifier
            baseClassifier_J48.setUseLaplace(true);
         
            
            BR learner1 = new BR();	  
            learner1.setClassifier(baseClassifier_J48);
            
            //Get the score of the training to use it to find the optimal threshold
            BR learner2 = new BR();
            Result trainR=new Result();
            
            int L=D_train.classIndex();
            int num=D_train.numInstances();
            double[][] trainscores=new double[num][L];
            int[][] trainactual=new int[num][L];

            D_test.setClassIndex(L);
            
            learner2.setClassifier(baseClassifier_J48);
            trainR=Evaluation.evaluateModel(learner2, D_train, D_train);
            
            trainscores=trainR.allPredictions();
            trainactual=trainR.allActuals();
            //End 

            //Run Experiment and print out the results;
            r=Evaluation.evaluateModel(learner1, D_train, D_test);
            
            //Get the result
            double [][] pred=r.allPredictions();
            int [][] actual=r.allActuals();
            Result.writeResultToFile(r,"/Users/ra12404/Desktop/meka-1.7.5/data/Thresholds/"+Mydataset+"/J48/"+"Results.txt");
            PrintWriter aa= new PrintWriter(new BufferedWriter(new FileWriter("/Users/ra12404/Desktop/meka-1.7.5/data/Thresholds/"+Mydataset+"/J48/"+"Scores.txt")));
         
            for(int i=0;i<pred.length;i++)
            {
            	for(int j=0;j<pred[0].length;j++)
            	{
            		aa.write(Double.toString(pred[i][j]));
            		aa.write(",");
            	}
            	aa.write("\n");
            }
            aa.close();

            HashMap<String,Double> stat=new LinkedHashMap<String,Double>() ;
            String output,t;
            double []FN;
            double[]FP;
            double[]CHL;
            double fn=0;
            double fp=0;
            double []tm=new double[L];
            double []sd =new double[L];
            //Total loss
            double sum=0,Loss=0;
            //Get L and N
            int N = pred.length;
            System.out.println("N==="+N);

            double card=MLUtils.labelCardinality(D_train,L);
            System.out.println(card);
            
            FN=new double[L];
            FP=new double[L];
            CHL=new double[L];
            tm=new double[L];
            double avg=0;
                     
            Random rand = new Random();
            //Defining the cost
            int a=0;
            int b=50;
            double bb=50.0;
            //Loss for each cost
            double []Q=new double[b+1]; 
            double []c =new double[b+1];  //for equal costs
            for (int i=0; i <= b; i++)
                c[i]=i/bb;
            
            double [][]cl=new double[b+1][L]; //for unequal costs
            for (int j=0; j < L; j++)
            {
                cl[a][j]=0;
                cl[b][j]=1;
            }
            
            for(int i=1;i<b;i++)
                for (int j=0; j < L; j++)
                    cl[i][j]= rand.nextDouble();
            
            //Apply different thresholds
            //a. Equal costs
            //1.Fixed score
            PrintWriter outer= new PrintWriter(new BufferedWriter(new FileWriter("/Users/ra12404/Desktop/meka-1.7.5/data/Thresholds/"+Mydataset+"/J48/"+"Fixed.txt")));
            String fname;
            Random d = new Random();           
            Loss=0;
            for (int i=0; i <= b; i++)
            {
                sum=0;fp=0;fn=0;
                t="0.5";
                stat=MLEvalUtils.getMLStats(pred, actual, t, "3","p");
                output=MLUtils.hashMapToString(stat,3);
                for(int j=0;j<L;j++)
                {
                    FN[j]=stat.get("FN["+j+"]");
                    FP[j]=stat.get("FP["+j+"]");
                    CHL[j]=2*(c[i]*(FN[j]/N)+(1-c[i])*(FP[j]/N));
                    sum+=CHL[j];
                    fn+=FN[j];
                    fp+=FP[j];
                }
                Q[i]=sum/L;
                fp=fp/(L*N);
                fn=fn/(L*N);
                Loss+=Q[i];
                outer.write(Double.toString(c[i]));
                outer.write("\t");
                outer.write(Double.toString(Q[i]));
                outer.write("\t");
                outer.write(Double.toString(fp));
                outer.write("\t");
                outer.write(Double.toString(fn));
                outer.write("\n");
            }
            Loss=Loss/(b+1);
            outer.close();
            //Write to file
            out.write("Error rate="+stat.get("Hamming loss"));
            out.write("\n\n");           
            out.write("Global Fixed");
            out.write("\n");
            out.write("Expected Loss="+Double.toString(Loss));
            out.write("\n");
            out.write("\n");
            //Print
            System.out.print("\n");
            System.out.println("Global Fixed");
            System.out.println("Expected Loss="+Loss);
            System.out.println("Error rate="+stat.get("Hamming loss"));
            //Put in results file
            writer.write(String.valueOf(df.format(Loss)));
            
            //2.Global PCut
            outer= new PrintWriter(new BufferedWriter(new FileWriter("/Users/ra12404/Desktop/meka-1.7.5/data/Thresholds/"+Mydataset+"/J48/"+"GlobalRateDriven.txt")));
            rand = new Random();
            sum=0;Loss=0;
            Q=new double[b+1];
            for (int i=0; i <= b; i++)
            {
                sum=0;fp=0;fn=0;
                
                t=Double.toString(ThresholdUtils.calibrateThreshold(r.predictions,card));
                stat=MLEvalUtils.getMLStats(pred, actual, t, "3");
                output=MLUtils.hashMapToString(stat,3);
                
                //Loss using uniform cost C
                for(int j=0;j<L;j++)
                {
                    FN[j]=stat.get("FN["+j+"]");
                    FP[j]=stat.get("FP["+j+"]");
                    CHL[j]=2*(c[i]*(FN[j]/N)+(1-c[i])*(FP[j]/N));
                    sum+=CHL[j];
                    fn+=FN[j];
                    fp+=FP[j];
                }
                Q[i]=sum/L;
                fp=fp/(L*N);
                fn=fn/(L*N);
                Loss+=Q[i];
                outer.write(Double.toString(c[i]));
                outer.write("\t");
                outer.write(Double.toString(Q[i]));
                outer.write("\t");
                outer.write(Double.toString(fp));
                outer.write("\t");
                outer.write(Double.toString(fn));
                outer.write("\n");
                
            }
            Loss=Loss/(b+1);
            outer.close();
            //Write to file
            out.write("Global Rate Driven");
            out.write("\n");
            out.write("Expected Loss="+Double.toString(Loss));
            out.write("\n");
            out.write("\n");
            //Print
            System.out.print("\n");
            System.out.println("Global Rate Driven");
            System.out.println("Expected Loss="+Loss);            
            //Put in results file
            writer.write(",");
            writer.write(String.valueOf(df.format(Loss)));
            
            //3.Global SD
            //Generate Random c between 0 and 1
            outer= new PrintWriter(new BufferedWriter(new FileWriter("/Users/ra12404/Desktop/meka-1.7.5/data/Thresholds/"+Mydataset+"/J48/"+"GlobalScoreDriven.txt")));
            double x;            
            sum=0;Loss=0;Q=new double[101];
            for (int i=0; i <= b; i++)
            {
                sum=0;fp=0;fn=0;
                //Threshold equals to the cost
                x=1-c[i];
                t=Double.toString(x);
                //Evaluate
                stat=MLEvalUtils.getMLStats(pred, actual, t, "3","p");
                output=MLUtils.hashMapToString(stat,3);
                //Loss using uniform cost C
                for(int j=0;j<L;j++)
                {
                    FN[j]=stat.get("FN["+j+"]");
                    FP[j]=stat.get("FP["+j+"]");
                    CHL[j]=2*(c[i]*(FN[j]/N)+(1-c[i])*(FP[j]/N));
                    sum+=CHL[j];
                    fn+=FN[j];
                    fp+=FP[j];
                }
                Q[i]=sum/L;
                fp=fp/(L*N);
                fn=fn/(L*N);
                Loss+=Q[i];
                outer.write(Double.toString(c[i]));
                outer.write("\t");
                outer.write(Double.toString(Q[i]));
                outer.write("\t");
                outer.write(Double.toString(fp));
                outer.write("\t");
                outer.write(Double.toString(fn));
                outer.write("\n");
            }
            Loss=Loss/(b+1);
            outer.close();
            //Write to file
            out.write("Global Score Driven ");
            out.write("\n");
            out.write("Expected Loss="+Double.toString(Loss));
            out.write("\n");
            out.write("\n");
            //Print
            System.out.print("\n");
            System.out.println("Global Score Driven");
            System.out.println("Expected Loss="+Loss);
            System.out.println("BR="+stat.get("Brier Score"));
            //Put in results file
            writer.write(",");
            writer.write(String.valueOf(df.format(Loss)));
            
            //4.Global SCut
            outer= new PrintWriter(new BufferedWriter(new FileWriter("/Users/ra12404/Desktop/meka-1.7.5/data/Thresholds/"+Mydataset+"/J48/"+"GlobalSCUT.txt")));
            rand = new Random();            
            sum=0;Loss=0;Q=new double[b+1];
            for (int i=0; i <= b; i++)
            {
                sum=0;fp=0;fn=0;
                
                for(int j=0;j<L;j++)
                    sd[j]=c[i];

                t=Double.toString(ThresholdUtils.calibrateThresholdSCut(trainscores,trainactual,c[i])); //here c is equal for all labels
                stat=MLEvalUtils.getMLStats(pred, actual, t, "3","p");
                output=MLUtils.hashMapToString(stat,3);
                //Loss using unifrom cost C
                for(int j=0;j<L;j++)
                {
                    FN[j]=stat.get("FN["+j+"]");
                    FP[j]=stat.get("FP["+j+"]");
                    CHL[j]=2*(c[i]*(FN[j]/N)+(1-c[i])*(FP[j]/N));
                    sum+=CHL[j];
                    fn+=FN[j];
                    fp+=FP[j];
                }
                
                Q[i]=sum/L;
                if(i==0 || i==b) Q[i]=0;
                fp=fp/(L*N);
                fn=fn/(L*N);
                Loss+=Q[i];
                outer.write(Double.toString(c[i]));
                outer.write("\t");
                outer.write(Double.toString(Q[i]));
                outer.write("\t");
                outer.write(Double.toString(fp));
                outer.write("\t");
                outer.write(Double.toString(fn));
                outer.write("\n");
                
            }
            Loss=Loss/(b+1);
            outer.close();
            //Write to file
            out.write("Global SCut");
            out.write("\n");
            out.write("Expected Loss="+Double.toString(Loss));
            out.write("\n");
            out.write("\n");
            //Print
            System.out.print("\n");
            System.out.println("Global SCut");
            System.out.println("Expected Loss="+Loss);            
            //Put in results file
            writer.write(",");
            writer.write(String.valueOf(df.format(Loss)));
            
            //5.L PCut
            outer= new PrintWriter(new BufferedWriter(new FileWriter("/Users/ra12404/Desktop/meka-1.7.5/data/Thresholds/"+Mydataset+"/J48/"+"LRateDriven.txt")));
            rand = new Random();                       
            sum=0;Loss=0;Q=new double[b+1];
            for (int i=0; i <= b; i++)
            {
                sum=0;fp=0;fn=0;
                //Find the threshold that achive a specific positive rate
                for(int j=0;j<L;j++)
                    sd[j]=c[i];
                
                t=Arrays.toString(ThresholdUtils.calibrateThresholds(r.predictions,sd));
                stat=MLEvalUtils.getMLStats(pred, actual, t, "3","p");
                output=MLUtils.hashMapToString(stat,3);
                
                //Loss using unifrom cost C
                for(int j=0;j<L;j++)
                {
                    FN[j]=stat.get("FN["+j+"]");
                    FP[j]=stat.get("FP["+j+"]");
                    CHL[j]=2*(c[i]*(FN[j]/N)+(1-c[i])*(FP[j]/N));
                    sum+=CHL[j];
                    fn+=FN[j];
                    fp+=FP[j];
                }
                Q[i]=sum/L;
                fp=fp/(L*N);
                fn=fn/(L*N);
                Loss+=Q[i];
                outer.write(Double.toString(c[i]));
                outer.write("\t");
                outer.write(Double.toString(Q[i]));
                outer.write("\t");
                outer.write(Double.toString(fp));
                outer.write("\t");
                outer.write(Double.toString(fn));
                outer.write("\n");
            }
            Loss=Loss/(b+1);
            outer.close();
            //Write to file
            out.write("L Rate Driven");
            out.write("\n");
            out.write("Expected Loss="+Double.toString(Loss));
            out.write("\n");
            out.write("\n");
            //Print
            System.out.print("\n");
            System.out.println("L Rate Driven ");
            System.out.println("Expected Loss="+Loss);
            //Put in results file
            writer.write(",");
            writer.write(String.valueOf(df.format(Loss)));
            
            //6.L SD
            outer=new PrintWriter(new BufferedWriter(new FileWriter("/Users/ra12404/Desktop/meka-1.7.5/data/Thresholds/"+Mydataset+"/J48/"+"LScoreDriven.txt")));
            rand = new Random();
            x=0;
            tm=new double[L];            
            sum=0;Loss=0;Q=new double[b+1];
            for (int i=0; i <= b; i++)
            {
                sum=0;x=0;avg=0;fn=0;fp=0;
                //Get the the cost per label
                for(int j=0;j<L;j++)
                    sd[j]=c[i];
                
                //Threshold equals to the cost
                for(int j=0;j<L;j++)
                    tm[j]=1-sd[j];
                
                t=Arrays.toString(tm);
                
                //Evaluate
                stat=MLEvalUtils.getMLStats(pred, actual, t, "3","p");
                output=MLUtils.hashMapToString(stat,3);
                //Loss using uniform cost C
                for(int j=0;j<L;j++)
                {
                    FN[j]=stat.get("FN["+j+"]");
                    FP[j]=stat.get("FP["+j+"]");
                    CHL[j]=2*(sd[j]*(FN[j]/N)+(1-sd[j])*(FP[j]/N));
                    sum+=CHL[j];
                    fn+=FN[j];
                    fp+=FP[j];
                }
                Q[i]=sum/L;
                Loss+=Q[i];
                outer.write(Double.toString(c[i]));
                outer.write("\t");
                outer.write(Double.toString(Q[i]));
                outer.write("\t");
                outer.write(Double.toString(fp));
                outer.write("\t");
                outer.write(Double.toString(fn));
                outer.write("\n");
            }
            Loss=Loss/(b+1);
            outer.close();
            //Write to file
            out.write("L Score Driven");
            out.write("\n");
            out.write("Expected Loss="+Double.toString(Loss));
            out.write("\n");
            out.write("\n");
            //Print
            System.out.print("\n");
            System.out.println("L Score Driven");
            System.out.println("Expected Loss="+Loss);
            //Put in results file
            writer.write(",");
            writer.write(String.valueOf(df.format(Loss)));
            
            //7.L SCut
            outer= new PrintWriter(new BufferedWriter(new FileWriter("/Users/ra12404/Desktop/meka-1.7.5/data/Thresholds/"+Mydataset+"/J48/"+"LSCUT.txt")));
            rand = new Random();            
            sum=0;Loss=0;Q=new double[b+1];
            for (int i=0; i <= b; i++)
            {
                sum=0;fp=0;fn=0;
                
                for(int j=0;j<L;j++)
                {
                    tm[j]=ThresholdUtils.calibrateThresholdsSCut(Mat.getCol(trainscores,j),Mat.getCol(trainactual,j),c[i]);
                }
                
                t=Arrays.toString(tm);
                stat=MLEvalUtils.getMLStats(pred, actual, t, "3","p");
                output=MLUtils.hashMapToString(stat,3);
                //Loss using unifrom cost C
                for(int j=0;j<L;j++)
                {
                    FN[j]=stat.get("FN["+j+"]");
                    FP[j]=stat.get("FP["+j+"]");
                    CHL[j]=2*(c[i]*(FN[j]/N)+(1-c[i])*(FP[j]/N));
                    sum+=CHL[j];
                    fn+=FN[j];
                    fp+=FP[j];
                }
                
                Q[i]=sum/L;
                if(i==0 || i==b) Q[i]=0;
                fp=fp/(L*N);
                fn=fn/(L*N);
                Loss+=Q[i];
                outer.write(Double.toString(c[i]));
                outer.write("\t");
                outer.write(Double.toString(Q[i]));
                outer.write("\t");
                outer.write(Double.toString(fp));
                outer.write("\t");
                outer.write(Double.toString(fn));
                outer.write("\n");
                
            }
            Loss=Loss/(b+1);
            outer.close();
            //Write to file
            out.write("L SCut ");
            out.write("\n");
            out.write("Expected Loss="+Double.toString(Loss));
            out.write("\n");
            out.write("\n");
            //Print
            System.out.print("\n");
            System.out.println("L SCut");
            System.out.println("Expected Loss="+Loss);            
            //Put in results file
            writer.write(",");
            writer.write(String.valueOf(df.format(Loss)));
            
            //8.L SDU
            double[]HL=new double[L];
            double[]BS=new double[L];
            double[]MAE=new double[L];
            double[]LossperLabel=new double[L];

            double exp_loss=0; //Total expected loss
            double hamming_loss=0;
            d=new Random();
            int max=L-1;
            int min=0;
            int rand_label=d.nextInt(max - min + 1) + min;
            double selected_score;
            Loss=0;
            System.out.println("Selected label="+(rand_label+1));
            System.out.println("\n");
            
	 		
            //Apply different thresholds
            outer= new PrintWriter(new BufferedWriter(new FileWriter("/Users/ra12404/Desktop/meka-1.7.5/data/Thresholds/"+Mydataset+"/J48/"+"OneRandomOthersFixed.txt")));

            for (int i=0; i <= b; i++)
            {     
            	sum=0;fp=0;fn=0;
                //Get the the cost for the rand_label 
                selected_score=c[i]; 
                
                for(int j=0;j<L;j++)
                    sd[j]=c[i];
                               
                for(int j=0;j<L;j++)
                	tm[j]=1-selected_score;                             	                        
                      
                t= Arrays.toString(tm);
                
                stat=MLEvalUtils.getMLStats(pred, actual, t, "3","p");
                output=MLUtils.hashMapToString(stat,3);
                for(int j=0;j<L;j++)
                {
                    FN[j]=stat.get("FN["+j+"]");
                    FP[j]=stat.get("FP["+j+"]");
                    CHL[j]=2*(sd[j]*(FN[j]/N)+(1-sd[j])*(FP[j]/N));
                    sum+=CHL[j];
                    fn+=FN[j];
                    fp+=FP[j];
                }
                Q[i]=sum/L;
                fp=fp/(L*N);
                fn=fn/(L*N);
                Loss+=Q[i];

                outer.write(Double.toString(c[i]));
                outer.write("\t");
                outer.write(Double.toString(Q[i]));
                outer.write("\t");
                outer.write(Double.toString(fp));
                outer.write("\t");
                outer.write(Double.toString(fn));
                outer.write("\n");
            }
            outer.close();
            //Put in results file
            Loss=Loss/(b+1);
            writer.write(",");
            writer.write(String.valueOf(df.format(Loss)));                    
		 	
            //9. RCut
            outer= new PrintWriter(new BufferedWriter(new FileWriter("/Users/ra12404/Desktop/meka-1.7.5/data/Thresholds/"+Mydataset+"/J48/"+"GlobalRCUT.txt")));
            rand = new Random();
            int k;
            sum=0;Loss=0;Q=new double[b+1];
            for (int i=0; i <= b; i++)
            {
                sum=0;fp=0;fn=0;
                
                //k=0 + (int)(Math.random()*(L+1));
                k=(int)Math.round(card);
                
                t=Arrays.toString(ThresholdUtils.calibrateThresholdsRCut(pred,k));
                stat=MLEvalUtils.getMLStats(pred, actual, t, "3", "r");
                output=MLUtils.hashMapToString(stat,3);
                //Loss using unifrom cost C
                for(int j=0;j<L;j++)
                {
                    FN[j]=stat.get("FN["+j+"]");
                    FP[j]=stat.get("FP["+j+"]");
                    CHL[j]=2*(c[i]*(FN[j]/N)+(1-c[i])*(FP[j]/N));
                    sum+=CHL[j];
                    fn+=FN[j];
                    fp+=FP[j];
                }

                Q[i]=sum/L;
                fp=fp/(L*N);
                fn=fn/(L*N);
                Loss+=Q[i];
                outer.write(Double.toString(c[i]));
                outer.write("\t");
                outer.write(Double.toString(Q[i]));
                outer.write("\t");
                outer.write(Double.toString(fp));
                outer.write("\t");
                outer.write(Double.toString(fn));
                outer.write("\n");
                
            }
            Loss=Loss/(b+1);
            outer.close();
            //Write to file
            out.write("Global RCut Driven ");
            out.write("\n");
            out.write("Expected Loss="+Double.toString(Loss));
            out.write("\n");
            out.write("\n");
            //Print
            System.out.print("\n");
            System.out.println("Global RCut Driven");
            System.out.println("Expected Loss="+Loss);            
            //Put in results file
            writer.write(",");
            writer.write(String.valueOf(df.format(Loss)));
            
            //10. MCut
            outer= new PrintWriter(new BufferedWriter(new FileWriter("/Users/ra12404/Desktop/meka-1.7.5/data/Thresholds/"+Mydataset+"/J48/"+"GlobalMCUT.txt")));
            rand = new Random();            
            sum=0;Loss=0;Q=new double[b+1];
            for (int i=0; i <= b; i++)
            {
                sum=0;fp=0;fn=0;
                
                t=Arrays.toString(ThresholdUtils.calibrateThresholdsMCut(pred));
                stat=MLEvalUtils.getMLStats(pred, actual, t, "3", "m");
                output=MLUtils.hashMapToString(stat,3);
                //Loss using unifrom cost C
                for(int j=0;j<L;j++)
                {
                    FN[j]=stat.get("FN["+j+"]");
                    FP[j]=stat.get("FP["+j+"]");
                    CHL[j]=2*(c[i]*(FN[j]/N)+(1-c[i])*(FP[j]/N));
                    sum+=CHL[j];
                    fn+=FN[j];
                    fp+=FP[j];
                }
                Q[i]=sum/L;
                fp=fp/(L*N);
                fn=fn/(L*N);
                Loss+=Q[i];
                outer.write(Double.toString(c[i]));
                outer.write("\t");
                outer.write(Double.toString(Q[i]));
                outer.write("\t");
                outer.write(Double.toString(fp));
                outer.write("\t");
                outer.write(Double.toString(fn));
                outer.write("\n");               
            }
            Loss=Loss/(b+1);
            outer.close();
            //Write to file
            out.write("Global MCut Driven ");
            out.write("\n");
            out.write("Expected Loss="+Double.toString(Loss));
            out.write("\n");
            out.write("\n");
            //Print
            System.out.print("\n");
            System.out.println("Global MCut Driven ");
            System.out.println("Expected Loss="+Loss);
            //Put in results file
            writer.write(",");
            writer.write(String.valueOf(df.format(Loss)));
                                   
            //***********************b. the unequal costs************************
            //1.Global Fixed
            outer= new PrintWriter(new BufferedWriter(new FileWriter("/Users/ra12404/Desktop/meka-1.7.5/data/Thresholds/"+Mydataset+"/J48/"+"Fixed_UQ.txt")));
            PrintWriter outerPerL= new PrintWriter(new BufferedWriter(new FileWriter("/Users/ra12404/Desktop/meka-1.7.5/data/Thresholds/"+Mydataset+"/J48/"+"PerLFixed_UQ.txt")));
            sd =new double[L];
            Loss=0;
            for (int i=0; i <= b; i++)
            {
                sum=0;avg=0;fn=0;fp=0;
                
                //Get the the cost per label
                for(int j=0;j<L;j++)
                    sd[j]=cl[i][j];
                
                t="0.5";
                stat=MLEvalUtils.getMLStats(pred, actual, t, "3","p");
                output=MLUtils.hashMapToString(stat,3);
                for(int j=0;j<L;j++)
                {
                    FN[j]=stat.get("FN["+j+"]");
                    FP[j]=stat.get("FP["+j+"]");
                    CHL[j]=2*(sd[j]*(FN[j]/N)+(1-sd[j])*(FP[j]/N));
                    avg+=sd[j];
                    sum+=CHL[j];
                    fn+=FN[j];
                    fp+=FP[j];
                    outerPerL.write(Double.toString(sd[j]));
                    outerPerL.write("\t");
                    outerPerL.write(Double.toString(CHL[j]));
                    outerPerL.write("\t");
                    outerPerL.write(Double.toString(FN[j]));
                    outerPerL.write("\t");
                    outerPerL.write(Double.toString(FP[j]));
                }
                outerPerL.write("\n");

                Q[i]=sum/L;
                fp=fp/(L*N);
                fn=fn/(L*N);
                Loss+=Q[i];
                avg=avg/L;
                outer.write(Double.toString(avg));
                outer.write("\t");
                outer.write(Double.toString(Q[i]));
                outer.write("\t");
                outer.write(Double.toString(fp));
                outer.write("\t");
                outer.write(Double.toString(fn));
                outer.write("\n");
            }
            Loss=Loss/(b+1);
            outer.close();
            outerPerL.close();
            //Write to file
            out.write("Global Fixed (Unequal Costs)");
            out.write("\n");
            out.write("Expected Loss="+Double.toString(Loss));
            out.write("\n");
            out.write("\n");
            //Print
            System.out.print("\n");
            System.out.println("Global Fixed (Unequal Costs)");
            System.out.println("Expected Loss="+Loss);  
            //Put in results file
            writer.write("\n");
            writer.write(",");
            writer.write(String.valueOf(df.format(Loss)));
            
            //2.Global PCut
            outer= new PrintWriter(new BufferedWriter(new FileWriter("/Users/ra12404/Desktop/meka-1.7.5/data/Thresholds/"+Mydataset+"/J48/"+"GlobalRateDriven_UQ.txt")));
            rand = new Random();
            double [] dc=new double[L];            
            sum=0;Loss=0;Q=new double[b+1];x=0;
            for (int i=0; i <= b; i++)
            {
                sum=0;x=0;fp=0;fn=0;
                //Get the the cost per label
                for(int j=0;j<L;j++)
                    sd[j]=cl[i][j];
                
                //Threshold equals to the cost
                for(int j=0;j<L;j++)
                    x+=sd[j];
                x=x/L;
                
                //Will setup the rate for all labels equal to x
                for(int j=0;j<L;j++)
                    dc[j]=x;
                
                //Find thresold that achive dc rate per label
                t=Double.toString(ThresholdUtils.calibrateThreshold(r.predictions,card));
                
                //System.out.println("T------"+t);
                stat=MLEvalUtils.getMLStats(pred, actual, t, "3");
                output=MLUtils.hashMapToString(stat,3);
                
                //Loss using unifrom cost C
                for(int j=0;j<L;j++)
                {
                    FN[j]=stat.get("FN["+j+"]");
                    FP[j]=stat.get("FP["+j+"]");
                    CHL[j]=2*(sd[j]*(FN[j]/N)+(1-sd[j])*(FP[j]/N));
                    sum+=CHL[j];
                    fn+=FN[j];
                    fp+=FP[j];
                }
                Q[i]=sum/L;
                fp=fp/(L*N);
                fn=fn/(L*N);
                Loss+=Q[i];
                outer.write(Double.toString(x));
                outer.write("\t");
                outer.write(Double.toString(Q[i]));
                outer.write("\t");
                outer.write(Double.toString(fp));
                outer.write("\t");
                outer.write(Double.toString(fn));
                outer.write("\n");
                
            }
            Loss=Loss/(b+1);
            outer.close();
            //Write to file
            out.write("Global Rate Driven (Unequal cost)");
            out.write("\n");
            out.write("Expected Loss="+Double.toString(Loss));
            out.write("\n");
            out.write("\n");
            //Print
            System.out.print("\n");
            System.out.println("Global Rate Driven (Unequal cost)");
            System.out.println("Expected Loss="+Loss);
            //Put in results file
            writer.write(",");
            writer.write(String.valueOf(df.format(Loss)));
            
            //3.Global SD
            outer= new PrintWriter(new BufferedWriter(new FileWriter("/Users/ra12404/Desktop/meka-1.7.5/data/Thresholds/"+Mydataset+"/J48/"+"GlobalScoreDriven_UQ.txt")));
            outerPerL= new PrintWriter(new BufferedWriter(new FileWriter("/Users/ra12404/Desktop/meka-1.7.5/data/Thresholds/"+Mydataset+"/J48/"+"PerLGlobalScoreDriven_UQ.txt")));
            rand = new Random();
            sd =new double[L];
            x=0;
                       
            sum=0;Loss=0;Q=new double[b+1];
            for (int i=0; i <= b; i++)
            {
                sum=0;x=0;avg=0;fn=0;fp=0;
                //Get the the cost per label
                for(int j=0;j<L;j++)
                    sd[j]=cl[i][j];
                //Threshold equals to the average cost
                for(int j=0;j<L;j++)
                    x+=sd[j];
                
                x=x/L;
                //Threshold equals to the average label cost
                x=1-x;
                t=Double.toString(x);
                //Evaluate
                stat=MLEvalUtils.getMLStats(pred, actual, t, "3","p");
                output=MLUtils.hashMapToString(stat,3);
                //Loss using unifrom cost C
                for(int j=0;j<L;j++)
                {
                    FN[j]=stat.get("FN["+j+"]");
                    FP[j]=stat.get("FP["+j+"]");
                    CHL[j]=2*(sd[j]*(FN[j]/N)+(1-sd[j])*(FP[j]/N));
                    sum+=CHL[j];
                    avg+=sd[j];
                    fn+=FN[j];
                    fp+=FP[j];
                    outerPerL.write(Double.toString(sd[j]));
                    outerPerL.write("\t");
                    outerPerL.write(Double.toString(CHL[j]));
                    outerPerL.write("\t");
                    outerPerL.write(Double.toString(FN[j]));
                    outerPerL.write("\t");
                    outerPerL.write(Double.toString(FP[j]));
                }
                outerPerL.write("\n");
                Q[i]=sum/L;
                Loss+=Q[i];
                fp=fp/(L*N);
                fn=fn/(L*N);
                avg=avg/L;
                outer.write(Double.toString(avg));
                outer.write("\t");
                outer.write(Double.toString(Q[i]));
                outer.write("\t");
                outer.write(Double.toString(fp));
                outer.write("\t");
                outer.write(Double.toString(fn));
                outer.write("\n");
            }
            Loss=Loss/(b+1);
            outer.close();
            outerPerL.close();
            //Write to file
            out.write("Global Score Driven (Unequal Costs)");
            out.write("\n");
            out.write("Expected Loss="+Double.toString(Loss));
            out.write("\n");
            out.write("\n");
            //Print
            System.out.print("\n");
            System.out.println("Global Score Driven (Unequal Costs)");
            System.out.println("Expected Loss="+Loss);
            //Put in results file
            writer.write(",");
            writer.write(String.valueOf(df.format(Loss)));
            
            //4.Global SCut
            outer= new PrintWriter(new BufferedWriter(new FileWriter("/Users/ra12404/Desktop/meka-1.7.5/data/Thresholds/"+Mydataset+"/J48/"+"GlobalSCUT_UQ.txt")));
            rand = new Random();
            
            sum=0;Loss=0;Q=new double[b+1];
            for (int i=0; i <= b; i++)
            {
                sum=0;fp=0;fn=0;avg=0;
                
                //Get the the cost per label
                for(int j=0;j<L;j++)
                    sd[j]=cl[i][j];
                
                for(int j=0;j<L;j++)
                    avg+=sd[j];
                
                avg=avg/L;
                                                      
                t=Double.toString(ThresholdUtils.calibrateThresholdSCut(trainscores,trainactual,avg));
                stat=MLEvalUtils.getMLStats(pred, actual, t, "3", "p");
                output=MLUtils.hashMapToString(stat,3);
                //Loss using unifrom cost C
                for(int j=0;j<L;j++)
                {
                    FN[j]=stat.get("FN["+j+"]");
                    FP[j]=stat.get("FP["+j+"]");
                    CHL[j]=2*(sd[j]*(FN[j]/N)+(1-sd[j])*(FP[j]/N));
                    sum+=CHL[j];
                    fn+=FN[j];
                    fp+=FP[j];
                }
                Q[i]=sum/L;
                fp=fp/(L*N);
                fn=fn/(L*N);
                Loss+=Q[i];
                outer.write(Double.toString(avg));
                outer.write("\t");
                outer.write(Double.toString(Q[i]));
                outer.write("\t");
                outer.write(Double.toString(fp));
                outer.write("\t");
                outer.write(Double.toString(fn));
                outer.write("\n");
                
            }
            Loss=Loss/(b+1);
            outer.close();
            //Write to file
            out.write("Global SCut (Unequal cost)");
            out.write("\n");
            out.write("Expected Loss="+Double.toString(Loss));
            out.write("\n");
            out.write("\n");
            //Print
            System.out.print("\n");
            System.out.println("Global SCut (Unequal cost)");
            System.out.println("Expected Loss="+Loss);
            //Put in results file
            writer.write(",");
            writer.write(String.valueOf(df.format(Loss)));
            
            //5.L PCut
            outer= new PrintWriter(new BufferedWriter(new FileWriter("/Users/ra12404/Desktop/meka-1.7.5/data/Thresholds/"+Mydataset+"/J48/"+"LRateDriven_UQ.txt")));
            rand = new Random();
            
            sum=0;Loss=0;Q=new double[b+1];
            for (int i=0; i <= b; i++)
            {
                sum=0;fp=0;fn=0;avg=0;
                //Find the threshold that achive a specific positive rate
                for(int j=0;j<L;j++)
                    sd[j]=cl[i][j];
                
                //Threshold equals to the cost
                for(int j=0;j<L;j++)
                    avg+=sd[j];
                avg=avg/L;
                
                t=Arrays.toString(ThresholdUtils.calibrateThresholds(r.predictions,sd));
                stat=MLEvalUtils.getMLStats(pred, actual, t, "3","p");
                output=MLUtils.hashMapToString(stat,3);
                
                //Loss using unifrom cost C
                for(int j=0;j<L;j++)
                {
                    FN[j]=stat.get("FN["+j+"]");
                    FP[j]=stat.get("FP["+j+"]");
                    CHL[j]=2*(sd[j]*(FN[j]/N)+(1-sd[j])*(FP[j]/N));
                    sum+=CHL[j];
                    fn+=FN[j];
                    fp+=FP[j];
                }
                Q[i]=sum/L;
                fp=fp/(L*N);
                fn=fn/(L*N);
                Loss+=Q[i];
                outer.write(Double.toString(avg));
                outer.write("\t");
                outer.write(Double.toString(Q[i]));
                outer.write("\t");
                outer.write(Double.toString(fp));
                outer.write("\t");
                outer.write(Double.toString(fn));
                outer.write("\n");
                
            }
            Loss=Loss/(b+1);
            outer.close();
            //Write to file
            out.write("L Rate Driven (Unequal cost)");
            out.write("\n");
            out.write("Expected Loss="+Double.toString(Loss));
            out.write("\n");
            out.write("\n");
            //Print
            System.out.print("\n");
            System.out.println("L Rate Driven (Unequal cost)");
            System.out.println("Expected Loss="+Loss);
            //Put in results file
            writer.write(",");
            writer.write(String.valueOf(df.format(Loss)));
            
            //6.L SD
            outer=new PrintWriter(new BufferedWriter(new FileWriter("/Users/ra12404/Desktop/meka-1.7.5/data/Thresholds/"+Mydataset+"/J48/"+"LScoreDriven_UQ.txt")));
            rand = new Random();
            sd =new double[L];
            x=0;
            tm=new double[L];
            
            sum=0;Loss=0;Q=new double[b+1];
            for (int i=0; i <= b; i++)
            {
                sum=0;x=0;avg=0;fn=0;fp=0;
                //Get the the cost per label
                for(int j=0;j<L;j++)
                    sd[j]=cl[i][j];
                
                //Threshold equals to the cost
                for(int j=0;j<L;j++)
                    tm[j]=1-sd[j];
                
                t=Arrays.toString(tm);
                
                //Evaluate
                stat=MLEvalUtils.getMLStats(pred, actual, t, "3","p");
                output=MLUtils.hashMapToString(stat,3);
                //Loss using unifrom cost C
                for(int j=0;j<L;j++)
                {
                    FN[j]=stat.get("FN["+j+"]");
                    FP[j]=stat.get("FP["+j+"]");
                    CHL[j]=2*(sd[j]*(FN[j]/N)+(1-sd[j])*(FP[j]/N));
                    avg+=sd[j];
                    sum+=CHL[j];
                    fn+=FN[j];
                    fp+=FP[j];
                }
                avg=avg/L;
                Q[i]=sum/L;
                Loss+=Q[i];
                outer.write(Double.toString(avg));
                outer.write("\t");
                outer.write(Double.toString(Q[i]));
                outer.write("\t");
                outer.write(Double.toString(fp));
                outer.write("\t");
                outer.write(Double.toString(fn));
                outer.write("\n");
            }
            Loss=Loss/(b+1);
            outer.close();
            //Write to file
            out.write("L Score Driven (Unequal cost)");
            out.write("\n");
            out.write("Expected Loss="+Double.toString(Loss));
            out.write("\n");
            out.write("\n");
            //Print
            System.out.print("\n");
            System.out.println("L Score Driven (Unequal cost)");
            System.out.println("Expected Loss="+Loss);
            //Put in results file
            writer.write(",");
            writer.write(String.valueOf(df.format(Loss)));
            
            //7.L SCut
            outer= new PrintWriter(new BufferedWriter(new FileWriter("/Users/ra12404/Desktop/meka-1.7.5/data/Thresholds/"+Mydataset+"/J48/"+"LSCUT_UQ.txt")));
            rand = new Random();
            
            sum=0;Loss=0;Q=new double[b+1];
            for (int i=0; i <= b; i++)
            {
                sum=0;fp=0;fn=0;avg=0;
                
                //Get the the cost per label
                for(int j=0;j<L;j++)
                    sd[j]=cl[i][j];
                
                for(int j=0;j<L;j++)
                    avg+=sd[j];
                
                avg=avg/L;
                
                for(int j=0;j<L;j++)
                {
                    tm[j]=ThresholdUtils.calibrateThresholdsSCut(Mat.getCol(trainscores,j),Mat.getCol(trainactual,j),sd[j]);
                }
                
                t=Arrays.toString(tm);
                stat=MLEvalUtils.getMLStats(pred, actual, t, "3","p");
                output=MLUtils.hashMapToString(stat,3);
                //Loss using unifrom cost C
                for(int j=0;j<L;j++)
                {
                    FN[j]=stat.get("FN["+j+"]");
                    FP[j]=stat.get("FP["+j+"]");
                    CHL[j]=2*(sd[j]*(FN[j]/N)+(1-sd[j])*(FP[j]/N));
                    sum+=CHL[j];
                    fn+=FN[j];
                    fp+=FP[j];
                }                
                Q[i]=sum/L;
                fp=fp/(L*N);
                fn=fn/(L*N);
                Loss+=Q[i];
                outer.write(Double.toString(avg));
                outer.write("\t");
                outer.write(Double.toString(Q[i]));
                outer.write("\t");
                outer.write(Double.toString(fp));
                outer.write("\t");
                outer.write(Double.toString(fn));
                outer.write("\n");
                
            }
            Loss=Loss/(b+1);
            outer.close();
            //Write to file
            out.write("L SCut (Unequal cost)");
            out.write("\n");
            out.write("Expected Loss="+Double.toString(Loss));
            out.write("\n");
            out.write("\n");
            out.close();
            //Print
            System.out.print("\n");
            System.out.println("L SCut (Unequal cost)");
            System.out.println("Expected Loss="+Loss);
            //Put in results file
            writer.write(",");
            writer.write(String.valueOf(df.format(Loss)));                      
            
            //8.L SDU
            HL=new double[L];
            BS=new double[L];
            MAE=new double[L];
            exp_loss=0; //Total expected loss
            hamming_loss=0;
            d=new Random();
            Loss=0;
            System.out.println("Selected label="+(rand_label+1));
            System.out.println("\n");
	 		
            outer= new PrintWriter(new BufferedWriter(new FileWriter("/Users/ra12404/Desktop/meka-1.7.5/data/Thresholds/"+Mydataset+"/J48/"+"OneRandomOthersFixed_UQ1.txt")));
          
            for (int i=0; i <= b; i++)
            {   sum=0;fp=0; fn=0;avg=0;            
                //Get the the cost for the rand_label 
                selected_score=cl[i][rand_label]; 
                
                for(int j=0;j<L;j++)
                    sd[j]=cl[i][j];
               
                for(int j=0;j<L;j++)
                    avg+=sd[j];
                
                avg=avg/L;
                
                for(int j=0;j<L;j++)
                	tm[j]=1-selected_score;                             	                        
                      
                t= Arrays.toString(tm);
                
                stat=MLEvalUtils.getMLStats(pred, actual, t, "3","p");
                output=MLUtils.hashMapToString(stat,3);
                for(int j=0;j<L;j++)
                {
                    FN[j]=stat.get("FN["+j+"]");
                    FP[j]=stat.get("FP["+j+"]");
                    CHL[j]=2*(sd[j]*(FN[j]/N)+(1-sd[j])*(FP[j]/N));
                    sum+=CHL[j];
                    fn+=FN[j];
                    fp+=FP[j];
                    LossperLabel[j]+=2*(sd[j]*(FN[j]/N)+(1-sd[j])*(FP[j]/N));
                    HL[j]+=stat.get("Hamming Loss["+j+"]");
                    BS[j]+=stat.get("Brier Score["+j+"]");
                    MAE[j]+=stat.get("MAE["+j+"]");
                }
                Q[i]=sum/L;
                fp=fp/(L*N);
                fn=fn/(L*N);
                Loss+=Q[i];
                outer.write(Double.toString(selected_score));
                outer.write("\t");
                outer.write(Double.toString(Q[i]));
                outer.write("\t");
                outer.write(Double.toString(fp));
                outer.write("\t");
                outer.write(Double.toString(fn));
                outer.write("\n");
            }
            outer.close();
            
            //Put in results file
            Loss=Loss/(b+1);
            writer.write(",");
            writer.write(String.valueOf(df.format(Loss)));

            //Print
            exp_loss=0;
            hamming_loss=0;
            double total_BS=0;
            double total_MAE=0;
            for(int j=0;j<L;j++)
            {
            	LossperLabel[j]=LossperLabel[j]/(b+1);
            	HL[j]=HL[j]/(b+1);
            	BS[j]=BS[j]/(b+1);
            	MAE[j]=MAE[j]/(b+1);
            	exp_loss+=LossperLabel[j];
            	hamming_loss+=HL[j];
            	total_BS+=BS[j];
            	total_MAE+=MAE[j];
            	System.out.println("Loss for label["+(j+1)+"]="+LossperLabel[j]);
                System.out.println("Hamming Loss for label["+(j+1)+"]="+HL[j]);
                System.out.println("Brier Score for label["+(j+1)+"]="+BS[j]);
                System.out.println("MAE for label["+(j+1)+"]="+MAE[j]);
                System.out.println("\n");
            }
            hamming_loss=hamming_loss/L;
            total_BS=total_BS/L;
            total_MAE=total_MAE/L;
            exp_loss=exp_loss/L;
            double f=((1.0/L)*(total_BS+((L-1)*total_MAE)));
            SDUout.write(Mydataset);
 	 		SDUout.write(',');
            SDUout.write(String.valueOf((rand_label+1)));
 	 		SDUout.write(',');
 	 		SDUout.write(String.valueOf(df.format(exp_loss)));
 		 	SDUout.write(',');
            SDUout.write(String.valueOf(df.format(total_BS)));
 	 		SDUout.write(',');
 	 		SDUout.write(String.valueOf(df.format(total_MAE)));
 	 		SDUout.write(',');
 	 		SDUout.write(String.valueOf(df.format(f)));
 		 	SDUout.write("\n");		

            
            //9. RCut
            outer= new PrintWriter(new BufferedWriter(new FileWriter("/Users/ra12404/Desktop/meka-1.7.5/data/Thresholds/"+Mydataset+"/J48/"+"GlobalRCUT_UQ.txt")));
            rand = new Random();
            
            sum=0;Loss=0;Q=new double[b+1];
            for (int i=0; i <= b; i++)
            {
                sum=0;fp=0;fn=0;avg=0;
                
                //k=0 + (int)(Math.random()*(L+1));
                k=(int)Math.round(card);

                //Get the the cost per label
                for(int j=0;j<L;j++)
                    sd[j]=cl[i][j];
                
                t=Arrays.toString(ThresholdUtils.calibrateThresholdsRCut(pred,k));
                stat=MLEvalUtils.getMLStats(pred, actual, t, "3", "r");
                output=MLUtils.hashMapToString(stat,3);
                //Loss using unifrom cost C
                for(int j=0;j<L;j++)
                {
                    FN[j]=stat.get("FN["+j+"]");
                    FP[j]=stat.get("FP["+j+"]");
                    CHL[j]=2*(sd[j]*(FN[j]/N)+(1-sd[j])*(FP[j]/N));
                    sum+=CHL[j];
                    avg+=sd[j];
                    fn+=FN[j];
                    fp+=FP[j];
                }
                Q[i]=sum/L;
                fp=fp/(L*N);
                fn=fn/(L*N);
                Loss+=Q[i];
                avg=avg/L;
                outer.write(Double.toString(avg));
                outer.write("\t");
                outer.write(Double.toString(Q[i]));
                outer.write("\t");
                outer.write(Double.toString(fp));
                outer.write("\t");
                outer.write(Double.toString(fn));
                outer.write("\n");
                
            }
            Loss=Loss/(b+1);
            outer.close();
            //Write to file
            out.write("Global RCut Driven (Unequal cost)");
            out.write("\n");
            out.write("Expected Loss="+Double.toString(Loss));
            out.write("\n");
            out.write("\n");
            //Print
            System.out.print("\n");
            System.out.println("Global RCut Driven (Unequal cost)");
            System.out.println("Expected Loss="+Loss);
            //Put in results file
            writer.write(",");
            writer.write(String.valueOf(df.format(Loss)));
            
            //10. MCut
            outer= new PrintWriter(new BufferedWriter(new FileWriter("/Users/ra12404/Desktop/meka-1.7.5/data/Thresholds/"+Mydataset+"/J48/"+"GlobalMCUT_UQ.txt")));
            rand = new Random();
            
            sum=0;Loss=0;Q=new double[b+1];
            for (int i=0; i <= b; i++)
            {
                sum=0;fp=0;fn=0;avg=0;
                
                //Get the the cost per label
                for(int j=0;j<L;j++)
                    sd[j]=cl[i][j];
                
                t=Arrays.toString(ThresholdUtils.calibrateThresholdsMCut(pred));
                stat=MLEvalUtils.getMLStats(pred, actual, t, "3", "m");
                output=MLUtils.hashMapToString(stat,3);
                //Loss using uniform cost C
                for(int j=0;j<L;j++)
                {
                    FN[j]=stat.get("FN["+j+"]");
                    FP[j]=stat.get("FP["+j+"]");
                    CHL[j]=2*(sd[j]*(FN[j]/N)+(1-sd[j])*(FP[j]/N));
                    sum+=CHL[j];
                    avg+=sd[j];
                    fn+=FN[j];
                    fp+=FP[j];
                }
                Q[i]=sum/L;
                fp=fp/(L*N);
                fn=fn/(L*N);
                Loss+=Q[i];
                avg=avg/L;
                outer.write(Double.toString(avg));
                outer.write("\t");
                outer.write(Double.toString(Q[i]));
                outer.write("\t");
                outer.write(Double.toString(fp));
                outer.write("\t");
                outer.write(Double.toString(fn));
                outer.write("\n");
                
            }
            Loss=Loss/(b+1);
            outer.close();
            //Write to file
            out.write("Global MCut Driven (Unequal cost)");
            out.write("\n");
            out.write("Expected Loss="+Double.toString(Loss));
            out.write("\n");
            out.write("\n");
            //Print
            System.out.print("\n");
            System.out.println("Global MCut Driven (Unequal cost)");
            System.out.println("Expected Loss="+Loss);            
            //Put in results file
            writer.write(",");
            writer.write(String.valueOf(df.format(Loss)));
            writer.write("\n\n");
           
	 		}
      	 writer.close();
		 SDUout.close(); 
        }
        catch (Exception ex)
        {
            Logger.getLogger(NewLoss.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
}