
/**
 * Implementation of LaCovaC: a decision tree classifier for multi-label based on LaCova algorithm. 
 * Al-Otaibi, RM, Kull, M & Flach, PA 2014, ‘Declaratively Capturing Local Label Correlations with Multi-Label Trees’. 
   in: The 22nd European Conference on Artificial Intelligence (ECAI-2016). 
 * IOS Press, pp.  1467-1475
 * @author Reem Al-Otaibi (ra12404@bristol.ac.uk)
 * @version August:2016 
 */
package meka.classifiers.multilabel;
import java.io.BufferedWriter;
import java.io.FileWriter;

import meka.core.MLEvalUtils;
import meka.core.MLUtils;
import meka.core.Result;
import weka.core.*;
import weka.core.converters.ConverterUtils.DataSource;
import meka.classifiers.multilabel.LaCovaC.*;

 @SuppressWarnings("serial")
public class LaCovaCClassifier extends MultilabelClassifier
{
  /** for serialization */ 
  private static final long serialVersionUID = 1L;

  /** The decision tree */
  protected LaCovaCTree m_root;

  /** Minimum number of instances */
  private int m_minNumObj ;
    
  protected BinC45ModelSelection Model;

   /**
   * Generates the classifier.
   * @param instances the data to train the classifier with
   * @throws Exception if classifier can't be built successfully
   */
  public void buildClassifier(Instances instances) throws Exception
  {
     testCapabilities(instances);
      
     m_minNumObj=5;

     Model=new BinC45ModelSelection(m_minNumObj,instances);

     m_root = new LaCovaCTree(Model);
    
     m_root.buildClassifier(instances,instances.classIndex());
     
     Model.cleanup();
   }
 
    /**
   * Returns class probabilities for an instance.
   * @param instance the instance to calculate the class probabilities for
   * @return the class probabilities
   * @throws Exception if distribution can't be computed successfully
   */
  public final double [] distributionForInstance(Instance instance) throws Exception
  {
    boolean m_useLaplace=true;
    double [] labels=m_root.distributionForInstance(instance, m_useLaplace);    
    return labels;
  }
  
    /**
   * Main method for testing this class
   *
   * @param argv the command line options
     * @throws Exception 
   */
  public static void main(String [] argv) throws Exception
 {
	   String path,trainingDataFilename,deploymentDataFilename;
	   DataSource sourcetrain, sourcedeploy;
	   Instances D_train, D_deploy; 
	   
       String datasetList[] ={"corel5k","CAL500","bibtex","LLOG","Enron","Birds","Medical","Genebase","SLASHDOT","Birds","Yeast","Flags","Emotions","Scene"};
	   path="/Users/ra12404/Desktop/meka-1.7.5/data/LaCova/Datasets/";
	   String Mydataset;
	   
	   BufferedWriter writer = new BufferedWriter(new FileWriter("/Users/ra12404/Desktop/meka-1.7.5/data/LaCova/Results/LaCovaC.csv"));

	 		writer.write("Dataset");
	 		writer.write(',');
	 		writer.write("Multi-label Accuracy");
	 		writer.write(',');
	 		writer.write("Exact-match");
	 		writer.write(',');
	 		writer.write("Error rate");
	 		writer.write(',');
	 		writer.write("F-measure");
	 		writer.write(',');
	 		writer.write("F1 macro avg by ex");
			writer.write(',');
			writer.write("F1 macro avg by lbl");
			writer.write(',');
			writer.write("Log Loss (max L)");
			writer.write(',');
			writer.write("Log Loss (max D)");
			writer.write(',');
			writer.write("Total time");
	 		writer.write('\n');		

	   /* using cx=cross validation, t=train-test, otherwise=terminal */
	   String option="cx";
	   	 
	   for (int d=0; d<datasetList.length; d++)
	   {	
		   Mydataset=datasetList[d];
		   writer.write(Mydataset);
	 	   writer.write(',');

	   /* using cross validation. */
	   if(option=="cx")
	   { 
		    Result r = null;
		    trainingDataFilename = path+Mydataset+".arff";
			sourcetrain = new DataSource(trainingDataFilename);
			D_train =  sourcetrain.getDataSet(); 
		    MLUtils.prepareData(D_train); 
		    
		    MultilabelClassifier cls=new  LaCovaCClassifier();

			Result[] folds=Evaluation.cvModel(cls, D_train, 10,"PCut1","3");
			
			r = MLEvalUtils.averageResults(folds);
			System.out.println(r.toString());   

			/* Writing to file */
			writer.write(r.info.get("Accuracy"));
			writer.write(',');
			writer.write(r.info.get("Exact match"));
			writer.write(',');
			writer.write(r.info.get("Hamming loss"));
			writer.write(',');
			writer.write(r.info.get("F1 micro avg"));
			writer.write(',');
			writer.write(r.info.get("F1 macro avg, by ex."));
			writer.write(',');
			writer.write(r.info.get("F1 macro avg, by lbl"));
			writer.write(',');
			writer.write(r.info.get("Log Loss (max L)"));
			writer.write(',');
			writer.write(r.info.get("Log Loss (max D)"));
			writer.write(',');
			writer.write(r.info.get("Total_time"));
			writer.write('\n');	
	   }
	   /* using train-test split. */
	   else if(option=="t")
	   {
		    Result r = null;
		    /* Training data */
		    trainingDataFilename = path+Mydataset+"-train.arff";
			sourcetrain = new DataSource(trainingDataFilename);
			D_train =  sourcetrain.getDataSet(); 
		    MLUtils.prepareData(D_train);
			
			/* Deployment data */
			deploymentDataFilename = path+Mydataset+"-test.arff";					
			sourcedeploy = new DataSource(deploymentDataFilename);
			D_deploy =  sourcedeploy.getDataSet(); 
		    MLUtils.prepareData(D_deploy); 
		    
		    MultilabelClassifier cls=new  LaCovaCClassifier();
			r=Evaluation.evaluateModel(cls, D_train, D_deploy, "PCut1", "3");
			System.out.println(r.toString());  			
	   }
	   /* using the terminal. argv are options passed through terminal. */
	   else
	   {
		   runClassifier(new LaCovaCClassifier(), argv);
	   }	
	   }
	   writer.flush();
	   writer.close();	  	 
  }
}

