package meka.classifiers.multilabel.ML45;

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
 *    BinC45Split.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *
 */
 
 /*
 *    Extended by Reem Alotaibi to handle multi-label data, 2016
 *	  ra12404@bristol.a.cuk
 */
import  weka.core.ContingencyTables;
import java.util.Enumeration;
import java.util.HashMap;
import weka.core.Instances;
import weka.core.Instance;
import weka.core.Utils;
import weka.core.RevisionUtils;
import meka.core.MLUtils;

import java.util.*;

import weka.core.matrix.*;

/**
 * Class implementing a binary C4.5-like split on an attribute.
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision: 8034 $
 */
public class BinC45Split extends ClassifierSplitModel
{
  /** for serialization */
  private static final long serialVersionUID = -1278776919563022474L;
    
  /** Attribute to split on. */
  private int m_attIndex;        

  /** Minimum number of objects in a split.   */ 
  private int m_minNoObj;            

  /** Value of split point. */
  private double m_splitPoint;

  /** The sum of the weights of the instances. */
  private double m_sumOfWeights;
    
    /** InfoGain of split. */
    private double m_infoGain;
    
    /** GainRatio of split.  */

    private double m_gainRatio;
    
    /** Static reference to splitting criterion. */
    private InfoGain CM=new InfoGain(); ;
    


  /**
   * Initializes the split model.
   */
  public BinC45Split(int attIndex,int minNoObj,double sumOfWeights)
  {
    // Get index of attribute to split on.
    m_attIndex = attIndex;
        
    // Set minimum number of objects.
    m_minNoObj = minNoObj;

    // Set sum of weights;
    m_sumOfWeights = sumOfWeights;

  }

  /**
   * Creates a C4.5-type split on the given data.
   *
   * @exception Exception if something goes wrong
   */
  public void buildClassifier(Instances trainInstances)throws Exception
 {

    // Initialize the remaining instance variables.
    m_numSubsets = 0;
    m_splitPoint = Double.MAX_VALUE;
    m_infoGain = 0;
    m_gainRatio = 0;

    // Different treatment for enumerated and numeric
    // attributes.
    if (trainInstances.attribute(m_attIndex).isNominal())
    {
      handleEnumeratedAttribute(trainInstances);
    }
    else
    {
      trainInstances.sort(trainInstances.attribute(m_attIndex));
      handleNumericAttribute(trainInstances);
    }
  }    

  /**
   * Returns index of attribute for which split was generated.
   */
  public final int attIndex()
  {
    return m_attIndex;
  }
  
  /**
   * Returns the split point (numeric attribute only).
   * 
   * @return the split point used for a test on a numeric attribute
   */
  public double splitPoint()
  {
    return m_splitPoint;
  }
  
    public final double infoGain()
    {
        
        return m_infoGain;
    }
    
    public final double gainRatio()
    {
        return m_gainRatio;
    }
    /**
   * Gets class probability for instance.
   *
   * @exception Exception if something goes wrong
   */
    public final double classProb(int l, int classIndex,Instance instance,int theSubset) throws Exception
    {
        Distribution d;
        if (theSubset <= -1)
        {
            double [] weights = weights(instance,l);
            if (weights == null)
            {
                d=bestDisMap.get(l);
                return d.prob(classIndex);
            }
            else
            {
                double prob = 0;
                d=bestDisMap.get(l);
                for (int i = 0; i < weights.length; i++)
                {
                    prob += weights[i] * d.prob(classIndex, i);
                }
                return prob;
            }
        }
        else
        {
            d=bestDisMap.get(l);

            if (Utils.gr(d.perBag(theSubset), 0))
            {
                return d.prob(classIndex, theSubset);
            }
            else
            {
                return d.prob(classIndex);
            }
        }
    }

  /**
   * Creates split on enumerated attribute.
   *
   * @exception Exception if something goes wrong
   */
    
    private void handleEnumeratedAttribute(Instances trainInstances)throws Exception
    {
        
        int L = trainInstances.classIndex();
        int N=trainInstances.numInstances();
        double []Q;
        double []n;
        int numvalue, bestf=-1,S0=0,S1=0;
        List<Double> thisLabel0=new ArrayList<Double>();
        List<Double> thisLabel1=new ArrayList<Double>();
        List<List<Double>> labelsListS0 = new ArrayList<List<Double>>();
        List<List<Double>> labelsListS1 = new ArrayList<List<Double>>();
        
        int numAttValues=0;
        numAttValues = trainInstances.numDistinctValues(m_attIndex);
        double []splitEntropy=new double[numAttValues];
        Q=new double[numAttValues];
        n=new double[numAttValues];       

        Distribution d=new Distribution(numAttValues,2);//bag for each nominal split, 2 for binary labels
        
        //Find labels distribution
        for(int j=0; j<L; j++)
        {
            thisLabel0.clear();
            thisLabel1.clear();
            S0=0;
            S1=0;
            d=new Distribution(numAttValues,2);
            for (int m=0; m<trainInstances.numInstances(); m++)
            {
                Instance x=trainInstances.instance(m);
                d.add((int)x.value(m_attIndex),x,j);
                
                if(x.value(m_attIndex)==0)
                {
                    thisLabel0.add(x.value(j));
                    S0+=1;
                }
                else
                {
                    thisLabel1.add(x.value(j));
                    S1+=1;
                }
            }
            labelsListS0.add(thisLabel0);
            labelsListS1.add(thisLabel1);
            bestDisMap.put(j,d);//m_distribution
        }
        
        if(S0>1 && S1>1)
            m_numSubsets=2;
                
        double currIG, currGR;
        
        //Generate Split for the att
        Instances [] instancesSplit;
        instancesSplit=split_instances(trainInstances,m_attIndex,numAttValues);
        
        HashMap<Integer, Distribution> secondDistribution = new HashMap<Integer, Distribution>();

        Distribution d2=new Distribution(numAttValues,2);//bag for each nominal split, 2 for binary labels
        
        for (int i = 0; i < instancesSplit.length; i++)
        {

            if (Utils.grOrEq(instancesSplit[i].numInstances(), m_minNoObj)) 
            {
              // Check if minimum number of Instances in the two subsets.
            	
                m_numSubsets = 2;
                currIG = CM.splitCritValue(trainInstances,instancesSplit,m_sumOfWeights);
                currGR = CM.gainRatio(trainInstances,instancesSplit,m_sumOfWeights, currIG);
                if ((i == 0) || Utils.gr(currGR, m_gainRatio)) 
                {
                  m_gainRatio = currGR;
                  m_infoGain = currIG;
                  m_splitPoint = i;
                  for(int l=0;l<L;l++)
                  {
                	 d2=new  Distribution(numAttValues,2);
                	 d2=bestDisMap.get(l);
                	 bestDisMap.put(l,new Distribution(d2, i));
                  }
                }              
            }
          }
        }   
    
    //Method to Split instances into two bins
    public final Instances [] split_instances(Instances data, int index, int num) throws Exception
    {
        
        Instances [] instances = new Instances [num];
        double [] weights;
        double newWeight;
        Instance instance;
        int subset, i, j;
        
        for (j=0;j<num;j++)
            instances[j] = new Instances((Instances)data,data.numInstances());
       
         for (i = 0; i < data.numInstances(); i++)
         {
            instance = ((Instances) data).instance(i);
                 if(instance.value(index)==0)
                     instances[0].add(instance);
                 else
                     instances[1].add(instance);
         }
       
        for (j = 0; j < num; j++)
            instances[j].compactify();
        
        return instances;
    }
  /**
   * Creates split on numeric attribute.
   *
   * @exception Exception if something goes wrong
   */
    private void handleNumericAttribute(Instances trainInstances)throws Exception
    {
    	int next = 1;
        int last = 0;
        int index = 0;
        int splitIndex = -1;
        double minSplit,total=0;

    	 int L = trainInstances.classIndex();
         int N=trainInstances.numInstances();
         int numAttValues;
         numAttValues = trainInstances.numDistinctValues(m_attIndex);
         
         double []n=new double[numAttValues];
         
         double []splitEntropy=new double[numAttValues];         
         
         Distribution d=new Distribution(2,2);// 2 for binary split
         
         double defaultEnt=0, currentInfoGain;
         
         //splitEntropyTotal=0, currGain=0;

         // Compute minimum number of Instances required in each subset.
         minSplit =  0.1*(d.total())/2.0;
         if (Utils.smOrEq(minSplit,m_minNoObj))
             minSplit = m_minNoObj;
         else
             if (Utils.gr(minSplit,25))
                 minSplit = 25;
         
         // Enough Instances with known values?
         if (Utils.sm((double)N,2*minSplit))
             return;
         
         Instances [] InstancesBags=new Instances[2] ;
         InstancesBags[0]=new Instances(trainInstances,0);
         InstancesBags[1]=new Instances(trainInstances);
         
         defaultEnt=CM.oldEnt(trainInstances, L);

       //we start by putting all instances in one bag and then move each two instances to check the split cut.
         while (next < N)
         {
             if (trainInstances.instance(next-1).value(m_attIndex)+1e-5 <trainInstances.instance(next).value(m_attIndex))
             {
                 
                 // Move all label-class values for all Instances up to next possible split point.
                     for(int i = last; i < next; i++)
                     {
                         InstancesBags[0].add(trainInstances.instance(i));
                         InstancesBags[1].delete(0);
                     }
                    
                     // Check if enough Instances in each subset and compute values for criteria.

                     if (Utils.grOrEq(InstancesBags[0].numInstances(),minSplit) && Utils.grOrEq(InstancesBags[1].numInstances(),minSplit))
                     {   
                    	 
                    	 currentInfoGain=CM.splitCritValue(trainInstances,InstancesBags,L,defaultEnt);
                    	 
                    	 if (Utils.gr(currentInfoGain, m_infoGain)) 
                    	 {
                             m_infoGain = currentInfoGain;
                             splitIndex = next - 1;
                           }                    	                        
                         index++;
                     }                         
                      last = next;
                     }
             next++;
             }    
         
         // Was there any useful split?
                       if (index == 0) 
                         return;
                       
                       if (Utils.smOrEq(m_infoGain, 0))
                       {                    	  
                    	      return;
                       }
                    // Set instance variables' values to values for best split.
                       m_splitPoint =(trainInstances.instance(splitIndex+1).value(m_attIndex)+trainInstances.instance(splitIndex).value(m_attIndex))/2;

                       // In case we have a numerical precision problem we need to choose the smaller value
                       if (m_splitPoint == trainInstances.instance(splitIndex + 1).value(m_attIndex))
                       {
                       m_splitPoint = trainInstances.instance(splitIndex).value(m_attIndex);
                       }

                       // Restore distribution for best split.
                       for(int j=0; j<L; j++)
                       {
                           d.addRange(0,trainInstances,0,splitIndex+1,j);
                           d.addRange(1,trainInstances,splitIndex+1,N,j);
                           bestDisMap.put(j,d);
                       }
                       //Restore the best split
                       Instances [] instancesSplit=new Instances[2];
                       //0 is the start index, splitIndex is number of instances to be copied
                       instancesSplit[0]=new Instances(trainInstances,0,splitIndex+1);
                    
                       //splitIndex is the start index, N-splitIndex is number of instances to be copied
                       instancesSplit[1]=new Instances(trainInstances,splitIndex+1,N-(splitIndex+1));
                       
                       if(instancesSplit[0].numInstances()>1 && instancesSplit[1].numInstances()>1)
                           m_numSubsets=2;
                       else 
                    	   return;
                       
                       m_gainRatio=CM.gainRatio(trainInstances,instancesSplit,m_sumOfWeights, m_infoGain);

}
        /**
   * Returns weights if instance is assigned to more than one subset.
   * Returns null if instance is only assigned to one subset.
   */
  public final double [] weights(Instance instance, int l)
 {    
    double [] weights;
    Distribution d;
    int i;
    d=bestDisMap.get(l);
     
    if (instance.isMissing(m_attIndex))
    {
      weights = new double [m_numSubsets];
      for (i=0;i<m_numSubsets;i++)
        weights [i] = d.perBag(i)/d.total();
      return weights;
    }
   else
    {
      return null;
    }
  }
  
  /**
   * Returns index of subset instance is assigned to.
   * Returns -1 if instance is assigned to more than one subset.
   *
   * @exception Exception if something goes wrong
   */

  public final int whichSubset(Instance instance) throws Exception
  {
    if (instance.isMissing(m_attIndex))
      return -1;
    else
    {
        if  (instance.attribute(m_attIndex).isNominal())
        {
            if ((int)m_splitPoint == (int)instance.value(m_attIndex))
                return 0;
            else
                return 1;

        }
        else
        {
            if (Utils.smOrEq(instance.value(m_attIndex),m_splitPoint))
                return 0;
            else
                return 1;
        }
     }
  }
  
  /**
   * Returns the revision string.
   * 
   * @return		the revision
   */
  public String getRevision()
  {
    return RevisionUtils.extract("$Revision: 8034 $");
  }
}
