
package meka.classifiers.multilabel.LaCovaC;
import java.util.ArrayList;

import weka.core.*;
import meka.core.MLUtils;

/**
 * Abstract class for computing covariance matrix and its statistic
 *
 * @author Reem Al-Otaibi (ra12404@bristol.ac.uk)
 */
public class CovarianceMatrix
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

  /**Find the sum of the pairwise covariances*/
      protected double find_covariance(Instances data, int L)
      {        

          double p[] = new double[L];
          double pw [][]=new double[L][L];
          double cov [][]=new double[L][L];
          double sum_cov=0;
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
                  cov[j][k]=Math.abs(pw[j][k]-(p[j]*p[k])); //pairwise covariance
                  
                  sum_cov+=cov[j][k]; //total sum of all pairwise covariance
                  
             }
              
          }         
          return sum_cov;
      }
    
      /**Find the upper triangle of the Covariance Matrix*/
      public double[][] find_CM(Instances data, int L)
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
                  cov[j][k]=Math.abs(pw[j][k]-(p[j]*p[k])); //pairwise covariance                                  
              }
              
          }         
          return cov;
      }
      
      /**Find the upper triangle+variance of the Covariance Matrix*/
      public double[][] find_CMV(Instances data, int L)
      {        

          double p[] = new double[L];
          double pw [][]=new double[L][L];
          double cov [][]=new double[L][L];
          int n=data.numInstances();
          
          p=labelFrequencies(data); //return the frequency of each label of data set.
          
          for(int j = 0; j < L; j++)
          {
              for(int k=j;k<L;k++)
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
              for(int k=j;k<L;k++)
              {
                  cov[j][k]=Math.abs(pw[j][k]-(p[j]*p[k])); //pairwise covariance                                  
              }
              
          }         
          return cov;
      }
    /**Find the Variance*/
    protected double find_variance(Instances data, int L)
    {
        double p[] = new double[L];
        double sum_var=0;
        
        p=labelFrequencies(data); //return the frequency of each label of data set.
        
        for(int j=0;j<L;j++)
        {
        	//System.out.println("p["+j+"]="+p[j]);
            sum_var+=p[j]*(1-p[j]);
        }
        
        return sum_var;
    }
    
    /**Estimate the threshold for the covariance*/  
    protected double find_thr_covariance(Instances data, int L)
    {
        double pi = 3.14;
        double thr;
        double sum1=0;
        double sum2=0;
        double mean=0;
        double sd=0;
        double p[] = new double[L];
        int n=data.numInstances();
        p=labelFrequencies(data); //return the frequency of each label of data set.
        
        for(int j=0;j<L;j++)
        {
            for(int k=j+1;k<L;k++)
            {
                 sum1+=Math.sqrt(p[j]*p[k]*(1-p[j])*(1-p[k]));
                 sum2+=p[j]*p[k]*(1-p[j])*(1-p[k]);
            }
            
        }       
        mean=Math.sqrt(2.0/(pi*(n-1)))*sum1;
        sd=Math.sqrt(((1-(2.0/pi))/(n-1))*sum2);
        thr=mean+2*sd; //confidence level

        return thr;
    }
    
    public double find_thr_pair_covariance(Instances data, int [] comb)
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
        
        for(int i = 0; i < n; i++) 
        {
            for(int j=0;j<comb.length;j++)
			{
            	labels[i][j]=(int)data.instance(i).value(comb[j]);
			}
            
		}
        //System.out.println("n="+n);
        for(int i=0;i<p.length;i++)
        	p[i]=MLUtils.labelCardinality(labels,i);//return the frequency of a given label of data set.
        
        int j=0;
        int k=1;

        sum1=Math.sqrt(p[j]*p[k]*(1-p[j])*(1-p[k]));
        sum2=p[j]*p[k]*(1-p[j])*(1-p[k]);
                   
        mean=Math.sqrt(2.0/(pi*(n-1)))*sum1;
        sd=Math.sqrt(((1-(2.0/pi))/(n-1))*sum2);
        thr=mean+2*sd; //confidence level

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
	
 }


