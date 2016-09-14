package weka.classifiers.trees.Verstile_11_3_2015;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.Arrays;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.REPTree;
import weka.core.AttributeStats;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.estimators.DiscreteEstimator;
/**
 * The Baseline3 that always test shift using KS and use linearly adapted threshold in case a shift is detected.
 * @author	Reem Al-Otaibi, ra12404@bristol.ac.uk
 * @version April 2015
 * See Al-Otaibi et al. "Versatile Decision Trees for Learning Over Multiple Contexts". ECML 2015
 * */
public class LinearTransformationKS 
{
    //Baseline 3 using alpha and beta KS
	public static void main(String[] args) throws Exception 
	{
		VersatileTree learner=new VersatileTree();
		double averageAcc=0;
		Evaluation eval ;
		int numFolds=5;
		int degreeShift=0,run;
		int maxrun=5;
		double avgAcc_shifts[][]=new double[maxrun][11];
		String mydataset="",path="";
		String datasetList[]={"appendicitis","bupa","spambase","threenorm","ringnorm","ion","pima","sonar","phoneme","breast-w"};
    	DiscreteEstimator m_ClassDistribution;
		String trainingDataFilename=null ;
		DataSource sourcetrain;
		Instances D_train ;  
		String deploymentDataFilename=null ;
		DataSource sourcedeploy;
		Instances D_deploy;		
		int i,j;
		BufferedWriter writer = new BufferedWriter(new FileWriter("/Users/ra12404/Desktop/weka-3-7-10/data/LinearKS_1_4_16.csv"));

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

						double[]probs = new double[D_deploy.numClasses()];
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
				        double p_value,alpha,beta,m1,o1,m2,o2,var,x,lng;
				        double []s1=null; double[]s2=null;
			        
			    	      // SK
					        for( i=0;i<D_deploy.numAttributes()-1;i++)
					        {
					        //values of the i-th attribute of training and deployment data
					        s1=D_train.attributeToDoubleArray(i);
					        s2=D_deploy.attributeToDoubleArray(i);

					        //Shift detect using KS test
					        p_value=VersatileModel.ksPValue(s1,s2);
					        	
					        //Accept H0 (s1=s2) if p-value>.05; otherwise, reject, meaning there is a shift
					       if (p_value<0.05)
					        {	  
					    	   	m1 = D_train.meanOrMode(i);    //mean
				        		var=D_train.variance(i);        //variance
				        		o1=Math.sqrt(var);           //standard deviation				        	    
				        	    m2 = D_deploy.meanOrMode(i);    //mean
				        		var=D_deploy.variance(i);        //variance
				        		o2=Math.sqrt(var);           //standard deviation				        	    
				        	    alpha=o2/o1;
				        	    beta=m2-alpha*m1;
				        	    //Correct the threshold
				        	    learner.ShiftedAtt(i);
						        learner.correct_threshold(i,alpha,beta);				        
					        }
					        }
	
						eval= new Evaluation(D_deploy);
						eval.evaluateModel(learner, D_deploy);
						averageAcc +=eval.pctCorrect();			
					}
						averageAcc=averageAcc/numFolds;
						avgAcc_shifts[run][degreeShift]=averageAcc;
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
							avg[i]+=avgAcc_shifts[j][i];
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
	
	static double getMean(double [] data)
    {
        double sum = 0.0;
        for(double a : data)
            sum += a;
        return sum/data.length;
    }

	static double getVariance(double [] data)
    {
        double mean = getMean(data);
        double temp = 0;
        for(double a :data)
            temp += (mean-a)*(mean-a);
        return temp/data.length;
    }

	static double getStdDev(double [] data)
    {
        return Math.sqrt(getVariance(data));
    }
}

