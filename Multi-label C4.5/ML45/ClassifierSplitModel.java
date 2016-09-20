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
 *    ClassifierSplitModel.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *
 */
 /*
 *    Extended by Reem Alotaibi to handle multi-label data, 2016
 *	  ra12404@bristol.a.cuk
 */
package meka.classifiers.multilabel.ML45;
import java.io.Serializable;
import weka.core.*;
import meka.core.MLUtils;
import  java.util.HashMap;

/** 
 * Abstract class for classification models that can be used 
 * recursively to split the data.
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision: 8034 $
 */
public abstract class ClassifierSplitModel implements Cloneable, Serializable, RevisionHandler
{

  /** for serialization */
  private static final long serialVersionUID = 4280730118393457457L;

  /** Distribution of all labels. */
    
  public HashMap<Integer, Distribution> bestDisMap = new HashMap<Integer, Distribution>();

  /** Number of created subsets. */
  protected int m_numSubsets;
    
  public int m_labels;
    
  public double var;
    
  public double  cov;

  public int bestatt;


  /**
   * Allows to clone a model (shallow copy).
   */
  public Object clone()
  {

    Object clone = null;
    try
      {
      clone = super.clone();
       }
    catch (CloneNotSupportedException e)
      {
      } 
    return clone;
  }

  /**
   * Builds the classifier split model for the given set of instances.
   *
   * @exception Exception if something goes wrong
   */
  public abstract void buildClassifier(Instances instances) throws Exception;
  
  /**
   * Checks if generated model is valid.
   */
  public final boolean checkModel()
  {
    if (m_numSubsets > 0)
      return true;
    else
      return false;
  }
  
  /**
   * Classifies a given instance.
   *
   * @exception Exception if something goes wrong
   */
 /* public final double classifyInstance(Instance instance)throws Exception
  {
      int theSubset;
      Distribution d;
      d=bestDisMap.get(l);
      
      theSubset = whichSubset(instance);
      if (theSubset > -1)
          return (double)d.maxClass(theSubset);
      else
          return (double)d.maxClass();
  }
*/
  /**
   * Gets class probability for instance.
   *
   * @exception Exception if something goes wrong
   */
  public double classProb(int l, int classIndex, Instance instance, int theSubset)
       throws Exception
    {
        Distribution d;
        d=bestDisMap.get(l);

    if (theSubset > -1)
    {
      return d.prob(classIndex,theSubset);
    }
    else
    {
      double [] weights = weights(instance,l);
      if (weights == null)
      {
          return d.prob(classIndex);
      }
      else
        {
        double prob = 0;
        for (int i = 0; i < weights.length; i++)
        {
          prob += weights[i] * d.prob(classIndex, i);
        }
        return prob;
      }
    }
  }

  /**
   * Gets class probability for instance.
   *
   * @exception Exception if something goes wrong
   */
  public double classProbLaplace(int l,int classIndex, Instance instance,
				 int theSubset) throws Exception
    {
        Distribution d;
        d=bestDisMap.get(l);
    if (theSubset > -1)
    {
      return d.laplaceProb(classIndex, theSubset);
    }
    else
    {
      double [] weights = weights(instance,l);
      if (weights == null)
      {
	return d.laplaceProb(classIndex);
      }
      else
      {
	double prob = 0;
	for (int i = 0; i < weights.length; i++)
    {
	  prob += weights[i] * d.laplaceProb(classIndex, i);
	}
	return prob;
      }
    }
  }

  /**
   * Returns coding costs of model. Returns 0 if not overwritten.
   */
  public double codingCost()
 {
    return 0;
  }

  /**
   * Returns the distribution of class values induced by the model.
   */
  public final Distribution distribution(int l)
 {
     Distribution d;
     d=bestDisMap.get(l);
    return d;
  }

   
  /**
   * Returns the number of created subsets for the split.
   */
  public final int numSubsets()
 {
    return m_numSubsets;
  }
  
    /**
   * Splits the given set of instances into subsets.
   *
   * @exception Exception if something goes wrong
   */
  public final Instances [] split(Instances data) throws Exception
    {

    Instances [] instances = new Instances [m_numSubsets];
    double [] weights;
    double newWeight;
    Instance instance;
    int subset, i, j;

    for (j=0;j<m_numSubsets;j++)
      instances[j] = new Instances((Instances)data,data.numInstances());
    
    for (i = 0; i < data.numInstances(); i++)
    {
      instance = ((Instances) data).instance(i);
      //weights = weights(instance);
      subset = whichSubset(instance);
      if (subset > -1)
      {
	   instances[subset].add(instance);
      }
    }
    for (j = 0; j < m_numSubsets; j++)
      instances[j].compactify();
    
    return instances;
  }

  /**
   * Returns weights if instance is assigned to more than one subset.
   * Returns null if instance is only assigned to one subset.
   */
  public abstract double [] weights(Instance instance, int l);
  
  /**
   * Returns index of subset instance is assigned to.
   * Returns -1 if instance is assigned to more than one subset.
   *
   * @exception Exception if something goes wrong
   */
  public abstract int whichSubset(Instance instance) throws Exception;
}





