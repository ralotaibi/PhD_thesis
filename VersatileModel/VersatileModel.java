package weka.classifiers.trees.Verstile_11_3_2015;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.Arrays;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.Verstile_10_3_2015.VM_REPTree_Linear;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.estimators.DiscreteEstimator;
/**
 * The Verstile Model (VM) that checks context changes: linear, non-linear and no change and decides type of threshold to be used.
 * @author	Reem Al-Otaibi, ra12404@bristol.ac.uk
 * @version April 2015
 * See Al-Otaibi et al. "Versatile Decision Trees for Learning Over Multiple Contexts". ECML 2015
 * */
public class VersatileModel 
{

	public static void main(String[] args) throws Exception 
	{
		VersatileTree learner=new VersatileTree();//Note that the percentile here is computed globally while we can do it locally.
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
		Instances D_deploy_corrected;
		int []total=new int[11];
		BufferedWriter writer = new BufferedWriter(new FileWriter("/Users/ra12404/Desktop/weka-3-7-10/data/VM_1_4_16.csv"));

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
			System.out.println("Dataset:"+mydataset);
			
			for(run=0 ;run<maxrun;run++)
			{
				System.out.println("Run = "+(run+1));
				path="/Users/ra12404/Desktop/weka-3-7-10/data/Datashift/"+mydataset+"/run"+(run+1);

				for(degreeShift=0;degreeShift<11;degreeShift++)//1: original, 2: nonlinear, 11: mixture, others:linear
				{
					averageAcc=0;
					System.out.println("Shift degree is "+(degreeShift+1));
		
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
					
			       // System.out.println("----------Delpoyment----------");
			       
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
			        D_deploy_corrected=new Instances(D_deploy);
				    
				     // "----------Decisions----------";
				    double p_value,alpha,beta,m1,o1,m2,o2,var,x,lng;
			        double []s1=null; double[]s2=null;
			       
		        	//System.out.println("Fold:"+(f+1));	
		        	
		        	//First SK
			        for( i=0;i<D_deploy.numAttributes()-1;i++)
			        {
			        	//values of the i-th attribute of training and deployment data
			        	s1=D_train.attributeToDoubleArray(i);
			        	s2=D_deploy.attributeToDoubleArray(i);

			        	//Shift detect using KS test
			        	p_value=ksPValue(s1,s2);
			        	
			        	//Accept H0 (s1=s2) if p-value>.05; otherwise, reject, meaning there is a shift
			        	if (p_value<0.05)
			        	{
			        		//Find alpha and beta from the training data
			        		m1 = D_train.meanOrMode(i);    //mean
			        		var=D_train.variance(i);        //variance
			        		o1=Math.sqrt(var);           //standard deviation			        	    
			        	    m2 = D_deploy.meanOrMode(i);    //mean
			        		var=D_deploy.variance(i);        //variance
			        		o2=Math.sqrt(var);           //standard deviation			        	    
			        	    alpha=o2/o1;
			        	    beta=m2-alpha*m1;

			        		//Correct the shifted data using x=(x'-beta)/alpha
			        	    lng=D_deploy.numInstances();
			        		for (int n=0; n<lng; n++)
			        		{
			        			Instance instance = D_deploy.instance(n);
			        			x=(instance.value(i)-beta)/alpha;
			        			D_deploy_corrected.instance(n).setValue(i, x);
			        		}			        		
			        			System.out.println("Att="+(i+1)+" is shifted according to first KS");
			        			total[degreeShift]+=1;
			        	    
			        	    	//Second SK
					        	//values of the i-th attribute of training and deployment data
					        	s1=D_train.attributeToDoubleArray(i);
					        	s2=D_deploy_corrected.attributeToDoubleArray(i);

					        	//Shift detect using KS test
					        	p_value=ksPValue(s1,s2);
					        	
					        	//Accept H0 (s1=s2) if p-value>.05; otherwise, reject, meaning there is a shift
					        	if (p_value<0.05)
					        	{
					        		//Use the percentile because the shift seems non-linear
					        		learner.ShiftedAtt(i);
							        learner.use_percentile(i);				        
					        	    System.out.println("Att="+(i+1)+" is shifted according to second KS");
					        	}
					        	else
					        	{
								    //Use the corrected threshold
					        		learner.ShiftedAtt(i);
							        learner.correct_threshold(i,alpha,beta);				        
					        	}			        	
			        	}
			        	else
			        	{	
			        	    System.out.println("Att="+(i+1)+" is not shifted");
			        	    
						    //Use the original model: training threshold
			        	}
			        }			        			       
				       eval= new Evaluation(D_deploy);
				       eval.evaluateModel(learner, D_deploy);
				       averageAcc +=eval.pctCorrect();       
				}
				averageAcc=averageAcc/numFolds;
				avgAcc_shifs[run][degreeShift]=averageAcc;		
			}	
	}
	writer.write(mydataset);
	System.out.println("-------------------------");
	System.out.println("Reults of Dataset="+mydataset);
	double avg[]=new double[11];
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
	for(int s=0;s<11;s++)
	{
		total[s]=total[s]/(maxrun*numFolds);
		System.out.println("Attribute detected as shifted in Degree "+(s+1)+"="+total[s]+"\n");
	}

	System.out.println("-------------------------");
}
writer.flush();
writer.close();
}
	/**
     * Calculates the p-value of the Kolmogorov-Smirnov test. 
     * @param s1 the training values for particular attribute.
     * @param s2 the deployment values for the same attribute.
     * @return the Kolmogorov-Smirnov p-value.
     */
    public static double ksPValue(double[] s1, double[] s2)
    {
        double maxD = ksStatistic2(s1, s2);
        int n = s1.length * s2.length / (s1.length + s2.length);
        double lambda = (Math.sqrt(n) + 0.12 + 0.11 / Math.sqrt(n)) * maxD;
        return Math.exp(-2 * lambda * lambda);
    }
	/**
     * Calculate the test statistic for the Kolmogorov-Smirnov test over the two
     * samples s1 and s2.
     * H0: s1=s2. Null hypothesis
     * H1: s1 not equal s2. The alternative hypothesis.
     * @param s1 the training values for particular attribute.
     * @param s2 the deployment values for the same attribute.
     * @return the KS test statistic.
     */
	public static double ksStatistic(double[] s1, double[] s2)
	{		
			final double[] sx = s1;
			final double[] sy = s2;
			Arrays.sort(sx);
	        Arrays.sort(sy);
	        int n1 = sx.length;
	        int n2 = sy.length;
	        int index1 = 0, index2 = 0;
	        double y1, y2, difference, frac1 = 0.0, frac2 = 0.0;
	        double maxDifference = 0.0;//is the maximum difference
	        while (index1 < n1 && index2 < n2)
	        {
	            while (index1 < n1 - 1 && Math.abs(sx[index1] - sx[index1 + 1]) < 1e-6)
	                    index1++;
	            
	            while (index2 < n2 - 1 && Math.abs(sy[index2] - sy[index2 + 1]) < 1e-6)
	                    index2++;
	            
	            y1 = sx[index1];
	            y2 = sy[index2];

	            if ( (y1 - y2) < 1e-6 )
	                    {frac1 = (double)++index1 / n1;	            //System.out.println("frac1="+frac1);
	                    }	              
	            
	            if ( (y2 - y1) < 1e-6)
	                    {frac2 = (double)++index2 / n2;  //System.out.println("frac2="+frac2);
	                    }
	                   	            
	            difference = Math.abs(frac2 - frac1);

	            if (difference  > maxDifference)
	                    maxDifference = difference;
	        }
	  return maxDifference;  
	}
	/**
     * Fast implementation of KS test
     **/
	public static double ksStatistic2(double[] s1, double[] s2)
	{
    // Copy and sort the sample arrays
    final double[] sx = s1;
    final double[] sy = s2;
    Arrays.sort(sx);
    Arrays.sort(sy);
    final int n = sx.length;
    final int m = sy.length;

    int rankX = 0;
    int rankY = 0;
    long curD = 0l;

    // Find the max difference between cdf_x and cdf_y
    long supD = 0l;
    do {
        double z = Double.compare(sx[rankX], sy[rankY]) <= 0 ? sx[rankX] : sy[rankY];
        while(rankX < n && Double.compare(sx[rankX], z) == 0) {
            rankX += 1;
            curD += m;
        }
        while(rankY < m && Double.compare(sy[rankY], z) == 0) {
            rankY += 1;
            curD -= n;
        }
        if (curD > supD) {
            supD = curD;
        }
        else if (-curD > supD) {
            supD = -curD;
        }
    } while(rankX < n && rankY < m);
    return supD;
	}
	
}
