
package meka.classifiers.multilabel.LaCovaC;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import weka.core.*;
import meka.core.MLUtils;

/**
 * Abstract class for computing covariance matrix and its statistic
 *
 * @author Reem Al-Otaibi (ra12404@bristol.ac.uk)
 */
public class CorrelationMatrix
{
    
    /** Find c_ij */
    protected double[][] find_cij(Instances data, int L)
    {
        double p[] = new double[L];
        double pw [][]=new double[L][L];
        int n=data.numInstances();
        
        p=labelFrequencies(data); //return the frequency of each label of data set data.

        for(int j = 0; j < L; j++)
        {
            for(int k=0;k<L;k++)
            {
                for(int i=0;i<n;i++)
                {
                    if((data.instance(i).value(j)==data.instance(i).value(k))&&(data.instance(i).value(k)==1)) //both are 1's
                    {
                        pw[j][k] +=1;
                    }
                }
            }
        }
        
        return pw;
    }
      /**Find the upper triangle of the Correlation Matrix*/
      public double[][] find_CorrMatrix(Instances data, int L)
      {       
          double p[] = new double[L];
          double pw [][]=new double[L][L];
          double cov [][]=new double[L][L];
          int n=data.numInstances();
          
          p=labelFrequencies(data); //return the frequency of each label of data set.
          
          for(int j = 0; j < L; j++)
          {
              for(int k=j+1;k<L;k++)
              {
                  for(int i=0;i<n;i++)
                  {
                      if((data.instance(i).value(j)==data.instance(i).value(k))&&(data.instance(i).value(k)==1)) //both are 1's
                      {
                          pw[j][k] +=1;
                      }
                  }
                  pw[j][k] /= n;
              }
          }   
          
          for(int j=0;j<L;j++)
          {
              for(int k=j+1;k<L;k++)
              {
            	  double var=Math.sqrt((p[j]*(1-p[j]))*(p[k]*(1-p[k])));
            	  if(var==0)
            		  var=1;
                  cov[j][k]=Math.abs((pw[j][k]-(p[j]*p[k]))/var); //pairwise correlation  
                  if(pw[j][k]==1)
                	  cov[j][k]=1;
                  /*if(Double.isNaN(cov[j][k] ))
            	  {
                	  System.out.println("NAN");
                	  cov[j][k]=1;
            	  }  */              
              }
              
          }         
          return cov;
      }
      
      /**Find the upper triangle of the Correlation Matrix*/
      public double find_CorrMatrix(int [][] labels, int L)
      {       
          double p[] = new double[L];
          double pw [][]=new double[L][L];
          double cov [][]=new double[L][L];
          int n=labels.length;
          double sum_cor=0;
          
  		  if(L==0)
  			L=1;
  		  
  		  for(int j = 0; j < L; j++) 
  		  {
  			for(int i = 0; i < n; i++) 
  			{
  				p[j] += labels[i][j];
  			}
  			p[j] /= n;
  		  }
          
          for(int j = 0; j < L; j++)
          {
              for(int k=j+1;k<L;k++)
              {
                  for(int i=0;i<n;i++)
                  {
                      if((labels[i][j]==labels[i][k])&&(labels[i][k]==1)) //both are 1's
                      {
                          pw[j][k] +=1;
                      }
                  }
                  pw[j][k] /= n;
              }
          }   
          
          for(int j=0;j<L;j++)
          {
              for(int k=j+1;k<L;k++)
              {
            	  double var=Math.sqrt((p[j]*(1-p[j]))*(p[k]*(1-p[k])));
            	  
                  cov[j][k]=((pw[j][k]-(p[j]*p[k]))/var); //pairwise correlation  
                  
                  if(Double.isNaN(cov[j][k] ))
                	  {
                	  cov[j][k]=1;break;
                	  }
                  else
                      sum_cor+=Math.abs(cov[j][k]);
              }
              
          }   
          return sum_cor;
      }
      /**Find the distance of the Correlation Matrix*/
      public double[][] find_distance(double [][] d, int L)
      { 
    	  for(int j=0;j<L;j++)
          {
              for(int k=j+1;k<L;k++)
              {
            	  d[j][k]=1- Math.abs(d[j][k]);
              }             
         }
    	  return d;
      }
     /** Estimate the correlation threshold for pair of labels */
    public double find_thr_pair_correlation(Instances data, int [] comb)
    {
        double pi = 3.14;
        double thr;
        double sum1=0;
        double sum2=0;
        double mean=0;
        double sd=0;
        int labels[][] = new int[data.numInstances()][comb.length];
        double p[] = new double[comb.length];
        int n=data.numInstances();
        
        /*for(int i = 0; i < n; i++) 
        {
            for(int j=0;j<comb.length;j++)
			{
            	labels[i][j]=(int)data.instance(i).value(comb[j]);
			}
            
		}*/

        //for(int i=0;i<p.length;i++)
        	//p[i]=MLUtils.labelCardinality(labels,i);//return the frequency of a given label of data set.
        
        int j=0;
        int k=1;

        //sum1=Math.sqrt(p[j]*p[k]*(1-p[j])*(1-p[k]));
        //sum2=p[j]*p[k]*(1-p[j])*(1-p[k]);
                   
        mean=Math.sqrt(2.0/(pi*(n-1)));
        sd=Math.sqrt(((1-(2.0/pi))/(n-1)));
        thr=(mean+2*sd); //confidence level

        //sd=1/(Math.sqrt(n-1));
        //thr=1.96*sd; //confidence level
        
        return thr;
    }
    
	public  double[] labelFrequencies(Instances D) 
	{

		int L = D.classIndex();
		if(L==0)
			L=1;
		double lc[] = new double[L];
		for(int j = 0; j < L; j++) 
		{
			for(int i = 0; i < D.numInstances(); i++) 
			{
				lc[j] += D.instance(i).value(j);
			}
			lc[j] /= D.numInstances();
		}
		return lc;
	}
	/** Find the threshold for pair of labels using bootstrap */	
	public double bootstrap_thr(Instances data,int L, int [] comb) throws IOException, ClassNotFoundException
	{
		double thr=0;
		Random r=new Random();
		HashMap<Integer,List  > map1=new HashMap<Integer,List>();
		int len =comb.length;
		int [][]labels=new int[data.numInstances()][len];
		int sample=1000;
		double [] corr=new double[sample];
		
		for(int i=0;i<len;i++) 
			for(int j=0;j<data.numInstances();j++) 
				labels[j][i]=(int) data.instance(j).value(comb[i]);
		
		List<Integer> label = null;
		for(int i=0;i<len;i++) 
		{
			 label=new ArrayList<Integer>();

			for(int j=0;j<data.numInstances();j++)    
				label.add((int) data.instance(j).value(comb[i]));
			
			map1.put(i, label);
		}
		
		 //System.out.println(map1.toString());
		
		for(int s=0;s<sample;s++)
		{
			HashMap<Integer,List  > map2=new HashMap<Integer,List>();
	
			//Perform shuffle
			for(int i=0;i<len;i++) 
			{
				List<Integer> l=map1.get(i);
				Collections.shuffle(l);
				map2.put(i,l);
			}
		
			int[][] arr = new int[data.numInstances()][len];

			for(int i=0;i<len;i++) 
			{
				List<Integer> l=new ArrayList<Integer>();
				l=map2.get(i);

				for(int j=0;j<data.numInstances();j++)    
					arr[j][i]=l.get(j);
			}
			
			corr[s]=find_CorrMatrix(arr, len);
		}

		double sum_mean=0;
		for(int i=0;i<sample;i++)
		   sum_mean += corr[i]; // this is the calculation for summing up all the values

		double mean = sum_mean / sample;
		
		double sum_sd=0;
		for(int i=0;i<sample;i++)
			   sum_sd += Math.pow((corr[i]-mean),2);
		
		double sd=Math.sqrt(sum_sd/sample);
		
		//Convert to z-score
		//double meanz=0.5*Math.log((1+mean)/(1-mean));
		//double sdz=1/Math.sqrt(data.numInstances()-3);
		//double thrz=1.96*sdz+meanz;
		//
		
		thr=mean+(2*sd);

		return thr;
	}
/** Find the threshold for all labels using bootstrap */	
public double bootstrap_thr(Instances data,int L) throws IOException, ClassNotFoundException
{
	double thr=0;
	Random r=new Random();
	HashMap<Integer,List  > map1=new HashMap<Integer,List>();
	int [][]labels=new int[data.numInstances()][L];
	int sample=1000;
	double [] corr=new double[sample];
	
	for(int i=0;i<L;i++) 
		for(int j=0;j<data.numInstances();j++) 
			labels[j][i]=(int) data.instance(j).value(i);
	
	List<Integer> label = null;
	for(int i=0;i<L;i++) 
	{
		 label=new ArrayList<Integer>();

		for(int j=0;j<data.numInstances();j++)    
			label.add((int) data.instance(j).value(i));
		
		map1.put(i, label);
	}
	
	for(int s=0;s<sample;s++)
	{
		HashMap<Integer,List  > map2=new HashMap<Integer,List>();

		//Perform shuffle
		for(int i=0;i<L;i++) 
		{
			List<Integer> l=map1.get(i);
			Collections.shuffle(l);
			map2.put(i,l);
		}
	
		int[][] arr = new int[data.numInstances()][L];

		for(int i=0;i<L;i++) 
		{
			List<Integer> l=new ArrayList<Integer>();
			l=map2.get(i);

			for(int j=0;j<data.numInstances();j++)    
				arr[j][i]=l.get(j);
		}
		corr[s]=find_CorrMatrix( arr, L);
	}

	double sum_mean=0;
	for(int i=0;i<sample;i++)
	   sum_mean += corr[i]; // this is the calculation for summing up all the values

	double mean = sum_mean / sample;
	
	double sum_sd=0;
	for(int i=0;i<sample;i++)
		   sum_sd += Math.pow((corr[i]-mean),2);
	
	double sd=Math.sqrt(sum_sd/sample);
	
	thr=mean+(2*sd);

	return thr;
}
 }


