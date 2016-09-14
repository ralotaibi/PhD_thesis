package weka.classifiers.trees.Verstile_11_3_2015;

	import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.Arrays;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.estimators.DiscreteEstimator;
/**
 * The Baseline5 that always assume non-linear shift use percentile threshold.
 * @author	Reem Al-Otaibi, ra12404@bristol.ac.uk
 * @version April 2015
 * See Al-Otaibi et al. "Versatile Decision Trees for Learning Over Multiple Contexts". ECML 2015
 * */
	public class Percentile 
	{
		//Baseline 5 using the percentile without KS

		public static void main(String[] args) throws Exception 
		{
			VersatileTree learner=new VersatileTree();
			double averageAcc=0;
			Evaluation eval ;
			int numFolds=5;
			int degreeShift,run;
			int maxrun=5,i,j;
			double avgAcc_shifs[][]=new double[maxrun][11];
	    	double []probs=null;
	    	DiscreteEstimator m_ClassDistribution;
	    	String mydataset="",path="";
	    	String datasetList[]={"appendicitis","bupa","spambase","threenorm","ringnorm","ion","pima","sonar","phoneme","breast-w"};
			String trainingDataFilename=null ;
			DataSource sourcetrain;
			Instances D_train ;  
			String deploymentDataFilename=null ;
			DataSource sourcedeploy;
			Instances D_deploy;		

			BufferedWriter writer = new BufferedWriter(new FileWriter("/Users/ra12404/Desktop/weka-3-7-10/data/Percentile_1_4_16.csv"));

			writer.write("Dataset");
			for( i=0;i<11;i++)
			{
				writer.write(',');
				writer.write(String.valueOf(i+1));
			}
			writer.write('\n');		
			
			for (int d=0; d<datasetList.length; d++)
			{	
			mydataset=datasetList[d];
			for(run=0 ;run<maxrun;run++)
			{
			    path="/Users/ra12404/Desktop/weka-3-7-10/data/Datashift/"+mydataset+"/run"+(run+1);

					for(degreeShift=0;degreeShift<11;degreeShift++)//1: original, 2: nonlinear, 11: mixture, others:linear
					{
						averageAcc=0;					
							for(int f = 0; f < numFolds; f++) 
							{
								learner=new VersatileTree();

						        //"----------Training----------";
								trainingDataFilename = path+"/LinearSh_"+(degreeShift+1)+"_tr"+(f+1)+".arff";					
								sourcetrain = new DataSource(trainingDataFilename);
								D_train =  sourcetrain.getDataSet(); 
								D_train.setClassIndex(D_train.numAttributes() - 1);
						        learner.buildClassifier(D_train);	
						        
							     //"----------Deployment----------";
								deploymentDataFilename = path+"/LinearSh_"+(degreeShift+1)+"_deploy"+(f+1)+".arff";
								sourcedeploy = new DataSource(deploymentDataFilename);
							    D_deploy =  sourcedeploy.getDataSet();
							    D_deploy.setClassIndex(D_deploy.numAttributes() - 1); 
														       
						        if(D_deploy.classAttribute().isNominal())
						        {
						        	 m_ClassDistribution = new DiscreteEstimator(D_deploy.numClasses(), true);
						        	 for(i=0;i<D_deploy.numInstances();i++)
						        	 { 
						        		 Instance x=D_deploy.instance(i);
						        	     m_ClassDistribution.addValue(x.classValue(),x.weight());
						        	 }
						        	    probs = new double[D_deploy.numClasses()];
						        	    for ( j = 0; j < D_deploy.numClasses(); j++) 
						        	    {
						        	      probs[j] = m_ClassDistribution.getProbability(j);
						        	    }			       
						        }						        
						        learner.deploy_info(D_deploy, probs);
						        for( i=0;i<D_deploy.numAttributes()-1;i++)
						        {				        	
						        		//Use the percentile because the shift seems non-linear
						        		learner.ShiftedAtt(i);
								        learner.use_percentile(i);				        
						        }			        			       
							       eval= new Evaluation(D_deploy);
							       eval.evaluateModel(learner, D_deploy);
							       averageAcc +=eval.pctCorrect();       
							}
							averageAcc=averageAcc/numFolds;
							avgAcc_shifs[run][degreeShift]=averageAcc;		
						}	
				}
			double avg[]=new double[11];
			System.out.println("-------------------------");
			System.out.println("Reults of Dataset="+mydataset);

			writer.write(mydataset);

				for( i=0;i<11;i++)
				{
					
						for( j=0;j<maxrun;j++)
						{
							avg[i]+=avgAcc_shifs[j][i];
						}
						avg[i]/=maxrun;
					System.out.println(" Shift is "+(i+1)+" average accuracy="+avg[i]);	
					writer.write(',');
					double re=avg[i]/100;
					writer.write(String.valueOf((double)Math.round(re * 1000) / 1000));
				}
				writer.write('\n');	
			System.out.println("-------------------------");
			}
			writer.flush();
			writer.close();
}
	}