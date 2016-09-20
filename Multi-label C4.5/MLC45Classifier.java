/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *    J48.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *
 */

 /*
 *    Extended by Reem Alotaibi to handle multi-label data, 2016
 *	  ra12404@bristol.a.cuk
 */
package meka.classifiers.multilabel;
import java.io.BufferedWriter;
import java.io.FileWriter;

import meka.classifiers.multilabel.ML45.*;
import meka.core.MLEvalUtils;
import meka.core.MLUtils;
import meka.core.Result;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;


 public class MLC45Classifier extends MultilabelClassifier
{
  /** for serialization */
  static final long serialVersionUID = -217733168393644444L;

  /** The decision tree */
  protected ClassifierTree m_root;

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

     m_root = new ClassifierTree(Model);
    
     m_root.buildClassifier(instances);
     
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
    return m_root.distributionForInstance(instance, m_useLaplace);
  }
    /**
   * Main method for testing this class
   *
   * @param argv the commandline options
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
	   
	   BufferedWriter writer = new BufferedWriter(new FileWriter("/Users/ra12404/Desktop/meka-1.7.5/data/LaCova/Results/MLC45.csv"));

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
	 		writer.write("F1 macro avg, by ex.");
			writer.write(',');
			writer.write("F1 macro avg, by lbl");
			writer.write(',');
			writer.write("Log Loss (max L)");
			writer.write(',');
			writer.write("Log Loss (max D)");
			writer.write(',');
	 		writer.write("Time");
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
		    
		    MultilabelClassifier cls=new  MLC45Classifier();

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
		    /* Training data */
		    Result r = null;
		   
		    path="/Users/ra12404/Desktop/meka-1.7.5/data/LaCova/Datasets/";
		    trainingDataFilename = path+Mydataset+".arff";
			sourcetrain = new DataSource(trainingDataFilename);
			D_train =  sourcetrain.getDataSet(); 
		    MLUtils.prepareData(D_train);
			
			/* Deployment data */
			deploymentDataFilename = path+Mydataset+".arff";					
			sourcedeploy = new DataSource(deploymentDataFilename);
			D_deploy =  sourcedeploy.getDataSet(); 
		    MLUtils.prepareData(D_deploy); 
		    
		    MultilabelClassifier cls=new  MLC45Classifier();
			r=Evaluation.evaluateModel(cls, D_train, D_deploy, "PCut1", "3");
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
	   /* using the terminal. argv are options passed through terminal. */
	   else
	   {
		   runClassifier(new MLC45Classifier(), argv);
	   }	  	
	   }
	   writer.flush();
	   writer.close();	  	 
  }
}

