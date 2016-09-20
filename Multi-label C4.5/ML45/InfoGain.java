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
 *    SplitCriterion.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *
 */
 /*
 *    Extended by Reem Alotaibi to handle multi-label data, 2016
 *	  ra12404@bristol.a.cuk
 */
package meka.classifiers.multilabel.ML45;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;

import java.io.Serializable;
import weka.core.*;
import meka.core.MLUtils;
import java.util.Queue;
import java.util.LinkedList;
import weka.filters.supervised.instance.*;

/**
 * Abstract class for computing splitting criteria
 * with respect to distributions of class values.
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision: 8034 $
 */
public class InfoGain 
{

    /** for serialization */
    private static final long serialVersionUID = 5490996638027101259L;
    private static double log2 = Math.log(2);
        
    /**Find the Entropy*/
    
    public final double oldEnt(Instances data, int L) 
    {

        double returnValue = 0;
        int j;
        double p[] = new double[L];
        p=MLUtils.labelCardinalities(data); //return the frequency of each label of dataset data.
        
        for (j=0;j<L;j++)
          returnValue = returnValue+lnFunc(p[j]);
        
        return (lnFunc(data.numInstances())-returnValue)/log2; 
      }
    
    public final double newEnt(Instances [] InstancesBags, int L) 
    {
        
        double returnValue = 0;
        int i,j;

        for (i=0;i<InstancesBags.length;i++)
        {
        	double p[] = new double[L];
            p=MLUtils.labelCardinalities(InstancesBags[i]); //return the frequency of each label of dataset data.
         
            for (j=0;j<L;j++)
            {       	  
        	  returnValue = returnValue+lnFunc(p[j]);
            }
        
          returnValue = returnValue-lnFunc(InstancesBags[i].numInstances());
        }
        return -(returnValue/log2);
      }
    
    public final double splitCritValue(Instances data, Instances [] InstancesBags, int L, double defaultEnt) 
    {

    	double numerator;

        numerator = (defaultEnt - newEnt(InstancesBags,L));

        // Splits with no gain are useless.
        if (Utils.eq(numerator, 0)) 
        {
          return 0;
        }

        return numerator / data.numInstances();
      }

    public final double splitCritValue(Instances data, Instances [] InstancesBags,  double totalNoInst) 
    {

        double numerator;

        numerator = (oldEnt(data,data.classIndex()) - newEnt(InstancesBags,data.classIndex()));

        // Splits with no gain are useless.
        if (Utils.eq(numerator, 0)) 
        {
          return 0;
        }

        return numerator / data.numInstances();
      }
    
    protected double entropy(Instances data, int L)
    {
        double p[] = new double[L];
        double sum=0;
        double returnValue = 0;

        p=MLUtils.labelCardinalities(data); //return the frequency of each label of dataset data.
        
        for (int i = 0; i < L; i++)
        {
            returnValue += lnFunc(p[i]);
            sum += p[i];
        }
        if (Utils.eq(sum, 0))
        {
            return 0;
        }
        else
        {
            return (returnValue + lnFunc(sum)) / (sum * log2);
        }

    }
    
    private static double lnFunc(double num)
    {
        
        if (num <= 0)
        {
            return 0;
        }
        else
        {
            return num * Math.log(num);
        }
    }

    public final double gainRatio(Instances data, Instances[] bags, double totalnoInst,
    	    double numerator) 
    {

        double denumerator;
        // Compute split info.
        denumerator = splitEnt(data,bags, totalnoInst);

        // Test if split is trivial.
        if (Utils.eq(denumerator, 0)) 
        {
          return 0;
        }
        denumerator = denumerator / totalnoInst;

        return numerator / denumerator;
   }
    
    private final double splitEnt(Instances data,Instances[] bags, double totalnoInst) 
    {

        double returnValue = 0;
        int i;
        int L=data.classIndex();
        
      
        if (Utils.gr(data.numInstances(), 0)) 
        {
          for (i = 0; i < bags.length; i++) 
          {
            returnValue = returnValue - lnFunc(bags[i].numInstances());
          }
          returnValue = returnValue + lnFunc(totalnoInst);
        }
        return returnValue /log2;
      }
   
  }


