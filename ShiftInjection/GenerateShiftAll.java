
package m2;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.Arrays;
import m2.basic.Dataset;
import m2.basic.Utils;
import m2.basic.Utils.*;
import java.util.Random;
import m2.basic.myRandomizer;
import m2.bias.Bias;
import m2.io.*;

/*
  * This Method generate shift for all numerical attributes while keeping nominal.
  * 10 degrees of linear shift which change, the mean, the variance, or both.
  * It generates non linear shift using the cube and then transform back to the same distribution.
  * Mixture shift is done by selecting subset of the attributes (1/3) to be linearly shifted, (1/3) non linear shift 
  * and the remaining (1/3) unshifted. 
 */

public class GenerateShiftAll 
{
	/**
	 * @param args
	 * @throws Exception 
	 */
	private static String datasetList[]={"appendicitis","bupa","spambase","threenorm","ringnorm","ion","pima","sonar","phoneme","breast-w"};
	private static String myDataset;
	
	public static void main(String[] args) throws Exception 
	{
	
		int numFolds=5;
		int maxrun=5;
		String[] args2 = new String[6];
		
		for (int i=0; i<datasetList.length; i++)
		{
			myDataset=datasetList[i];
			System.out.println("Dataset:"+myDataset);
			for(int run=0 ;run<maxrun;run++)
			{
			args2[0] = "-dataset";
			args2[1] = "/Users/ra12404/Desktop/weka-3-7-10/data/Datashift/"+myDataset+".arff";
			args2[2] = "-folder";
			args2[3] = "/Users/ra12404/Desktop/weka-3-7-10/data/Datashift/"+myDataset+"/run"+(run+1);

			args = args2;

			String fileIn = Utils.getOption("dataset", args);
			String folder = Utils.getOption("folder", args);
			String biasType = "";
	
			//Read input data set
			Dataset original = FileFormat.createDatasetReader(fileIn).read();
			Dataset backup = original.clone();
			Dataset[] training;
			Dataset[] deployment;
			Dataset[] deploymentbeforeshift;

			Dataset[] backupTr = new Dataset[numFolds];
			Dataset[] backupTst = new Dataset[numFolds];
			Dataset[] backupTstOrg = new Dataset[numFolds];

			int attribute;
			Bias shift;
			double degreeShift;
			
			//Split the data set into training and test.
			original = backup.clone();
			original.randomize();
			original.createPartitions();
			training = original.getTrainingCV(numFolds);
			deploymentbeforeshift = original.getTestCV(numFolds);
			deployment=deploymentbeforeshift;
		
			for (int f=0; f<numFolds; f++)
			{
				backupTr[f] = training[f].clone();
				backupTstOrg[f]=deploymentbeforeshift[f].clone();
			}
			
			double gama=0,phi=0;
			//Inject linear shift for all attributes
			for (int j =1; j<11; j++)
			{
				if(j!=2)
				{

				if (j==1)
				{
					phi=0;gama=0;degreeShift=1;//no shift
				}
					//degreeShift=2;//non linear shift
				else if (j==3)
				{
					phi=0;gama=1;degreeShift=3;//mean shift
				}
				else if (j==4)
				{
					phi=0;gama=-1;degreeShift=4;//mean shift
				}
				else if (j==5)
				{
					phi=1;gama=0;degreeShift=5;//var shift
				}
				else if (j==6)
				{
					phi=-1;gama=0;degreeShift=6;//var shift
				}
				else if (j==7)
				{
					phi=1;gama=1;degreeShift=7;//mean and variance
				}
				else if (j==8)
				{
					phi=-1;gama=-1;degreeShift = 8;//mean and variance
				}
				else if (j==9)
				{
					phi=-1;gama=1;degreeShift = 9;//mean and variance
				}
				else  if(j==10)
				{
					phi=1;gama=-1;degreeShift = 10;//mean and variance
				}
				else
				{
					phi=0;gama=0;degreeShift=11;//mixture shift
				}
				
				System.out.println("j="+j+" phi="+phi+" gama="+gama);
				
				for (int f=0; f<numFolds; f++)
				{
					training[f] = backupTr[f].clone();
					deploymentbeforeshift[f]=backupTstOrg[f].clone();
				}
				
				 //Inject linear shift to all attributes
				biasType = "m2.bias.NewLinearShift";
				shift = (Bias)(Class.forName(biasType).newInstance());
				shift.setValue(phi,gama,degreeShift);
				
				for (int f=0; f<numFolds;f++)
				{
					String trFileName = folder+"/Original_"+j+"_deploy"+(f+1)+".arff";
					Dataset org = deploymentbeforeshift[f].clone();
		            PrintWriter outer= new PrintWriter(new BufferedWriter(new FileWriter(trFileName)));
		            ArffWriter arff=new ArffWriter(org,outer);
					arff.write();
		            outer.close();
				}

				for(int k=0;k<original.numAttributes()-1;k++)
				{		
					attribute=k;
					
					for (int f=0;f<numFolds;f++)
					{
						if (original.getAttribute(attribute).isNumeric())
							deployment[f] = shift.injectBias(deploymentbeforeshift[f], attribute);
					}
				}
				
				//save Data sets
				for (int f=0; f<numFolds;f++)
				{
					String trFileName = folder+"/LinearSh_"+j+"_tr"+(f+1)+".arff";
					Dataset trDataset = training[f].clone();
		            PrintWriter outer= new PrintWriter(new BufferedWriter(new FileWriter(trFileName)));
		            ArffWriter arff=new ArffWriter(trDataset,outer);
					arff.write();
		            outer.close();
		            
					String tstFileName = folder+"/LinearSh_"+j+"_deploy"+(f+1)+".arff";
		            outer= new PrintWriter(new BufferedWriter(new FileWriter(tstFileName)));
					Dataset tsDataset = deployment[f].clone();
				    arff=new ArffWriter(tsDataset,outer);
					arff.write();
				}
				}//end if j!=2		
			}//end linear loop options
		
			    //Inject non-linear shift for all attributes
				biasType = "m2.bias.NewNonLinearShift";
				shift = (Bias)(Class.forName(biasType).newInstance());				
				
				for (int f=0; f<numFolds; f++)
				{
					training[f] = backupTr[f].clone();
					deploymentbeforeshift[f]=backupTstOrg[f].clone();
				}			

				for(int k=0;k<original.numAttributes()-1;k++)
				{		
					attribute=k;
					for (int f=0;f<numFolds;f++)
					{
						if (original.getAttribute(attribute).isNumeric())
							deployment[f] = shift.injectBias(deploymentbeforeshift[f], attribute);
					}
				}				
				//save Data sets
				for (int f=0; f<numFolds;f++)
				{	
					String trFileName = folder+"/LinearSh_2_"+"tr"+(f+1)+".arff";
					Dataset trDataset = training[f].clone();
		            PrintWriter outer= new PrintWriter(new BufferedWriter(new FileWriter(trFileName)));
		            ArffWriter arff=new ArffWriter(trDataset,outer);
					arff.write();
		            outer.close();

					String tstFileName = folder+"/LinearSh_2_"+"deploy"+(f+1)+".arff";
		            outer= new PrintWriter(new BufferedWriter(new FileWriter(tstFileName)));
					Dataset tsDataset = deployment[f].clone();
				    arff=new ArffWriter(tsDataset,outer);
					arff.write();
				}
					
				//Inject mixture shift for all attributes
				int limit=2*(original.numAttributes()/3);
				int [] RandomAtt=new int[limit];
				int count=0;
				while(count < limit)
				{
				  int randomnumber=myRandomizer.generator.nextInt(original.numAttributes()-1);
				  boolean found1=false;				  
				  boolean found2=false;
				  for(int k=0;k<=count;k++)
				  {
				    if(RandomAtt[k]==randomnumber)
				     {
				        found1=true;
				    	break;
				    }
				    if(original.getAttribute(randomnumber).isNominal()|| original.getMin(randomnumber)==original.getMax(randomnumber))
				    {
				    	found2=true;
				    	break;	
				    }
				  }
				  if((!found1)&&(!found2))
				  {
					  RandomAtt[count]=randomnumber;
					  count=count+1;
				  }
				}
				Arrays.sort(RandomAtt);
				biasType = "m2.bias.NewLinearShift";
				shift = (Bias)(Class.forName(biasType).newInstance());
				
                //Inject linear shift for some attributes
				System.out.println("Numebr of attribute will be shifted in one-third option="+limit);

				for (int f=0; f<numFolds; f++)
				{
					training[f] = backupTr[f].clone();
					deploymentbeforeshift[f]=backupTstOrg[f].clone();
				}
								
				for(int k=0;k<RandomAtt.length/2;k++)
				{		
					attribute=RandomAtt[k];
					phi=0;gama=0;
					//Generating random gama and alpha
					while (gama==0)
					{
						Random rn1 = new Random();
						gama=rn1.nextInt((1-(-1)) + 1) + (-1);
					}
					while (phi==0)
					{
						Random rn2 = new Random();
						phi=rn2.nextInt((1 - (-1)) + 1) + (-1);
					}
					//
					shift.setValue(phi,gama,11);
					System.out.println(" phi="+phi+"and gama="+gama);
					System.out.println(" att selected="+attribute);

					for (int f=0;f<numFolds;f++)
					   {
							deployment[f] = shift.injectBias(deploymentbeforeshift[f], attribute);
					   }
				}
				
				biasType = "m2.bias.NewNonLinearShift";
				shift = (Bias)(Class.forName(biasType).newInstance());				
				
				for(int k=RandomAtt.length/2;k<RandomAtt.length;k++)
				{		
					attribute=RandomAtt[k];
					System.out.println(" att selected="+attribute);

					  for (int f=0;f<numFolds;f++)
					   {
							deployment[f] = shift.injectBias(deploymentbeforeshift[f], attribute);
					   }
				}
				
				//save Data sets
				for (int f=0; f<numFolds;f++)
				{
					String trFileName = folder+"/LinearSh_11"+"_tr"+(f+1)+".arff";
					Dataset trDataset = training[f].clone();
		            PrintWriter outer= new PrintWriter(new BufferedWriter(new FileWriter(trFileName)));
		            ArffWriter arff=new ArffWriter(trDataset,outer);
					arff.write();
		            outer.close();
		            
					String tstFileName = folder+"/LinearSh_11"+"_deploy"+(f+1)+".arff";
		            outer= new PrintWriter(new BufferedWriter(new FileWriter(tstFileName)));
					Dataset tsDataset = deployment[f].clone();
				    arff=new ArffWriter(tsDataset,outer);
					arff.write();
					outer.close();
				}		
			}
			}
		}//End Main

	}//End class
