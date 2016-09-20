
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
 *    ClassifierTree.java
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
import weka.classifiers.functions.SMO;

import java.io.Serializable;

import weka.core.*;
import meka.core.MLUtils;

import java.util.Queue;
import java.util.LinkedList;

import weka.filters.supervised.instance.*;


/**
 * Class for handling a tree structure used for
 * classification.
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision: 9117 $
 */
public class ClassifierTree implements  CapabilitiesHandler
{

  /** for serialization */
  static final long serialVersionUID = -8722249377542734193L;
  
  /** The model selection method. */  
  protected BinC45ModelSelection m_toSelectModel;     

  /** Local model at node. */
  protected ClassifierSplitModel m_localModel;  

  /** References to sons. */
  protected ClassifierTree [] m_sons;           

  /** True if node is leaf. */
  protected boolean m_isLeaf;
    
  /** True if node is single-label leaf. */
  protected boolean m_isSingleLabel;

  /** True if node is empty. */
  protected boolean m_isEmpty;                  

  /** The training instances. */
  protected Instances m_train;                  

  /** The pruning instances. */
  protected Distribution m_test;     

  /** The id for the node. */
  protected int m_id;
    
  /** The minimum number of instances at leaf. */

  protected int n;
    
  /**
   * Constructor. 
   */
    public ClassifierTree(BinC45ModelSelection toSelectLocModel)
    {
     m_toSelectModel = toSelectLocModel;
    }
    
    public ClassifierTree(BinC45ModelSelection toSelectLocModel,boolean status)
    {
        m_toSelectModel = toSelectLocModel;
        m_isSingleLabel=status;
    }

  /**
   * Returns default capabilities of the classifier tree.
   *
   * @return      the capabilities of this classifier tree
   */
  public Capabilities getCapabilities()
 {
    Capabilities result = new Capabilities(this);
    result.enableAll();
    return result;
  }

  /**
   * Method for building a classifier tree.
   *
   * @param data the data to build the tree from
   * @throws Exception if something goes wrong
   */
  public void buildClassifier(Instances data) throws Exception
  {
    int L=data.classIndex();
    // can classifier tree handle the data?
    getCapabilities().testWithFail(data);
    // remove instances with missing class
    data = new Instances(data);
    data.deleteWithMissingClass();
    buildTree(data, false,L);
    cleanup(new Instances(data, 0));
  }

  /**
   * Builds the tree structure.
   *
   * @param data the data for which the tree structure is to be
   * generated.
   * @param keepData is training data to be kept?
   * @throws Exception if something goes wrong
   */
  public void buildTree(Instances data, boolean keepData, int labels) throws Exception
    {

    Instances [] localInstances;

    if (keepData)
    {
      m_train = data;
    }
        
    int L = data.classIndex();

    m_test = null;
    m_isLeaf = false;
    m_isEmpty = false;
    m_sons = null;
    
	//Find best split model
    m_localModel = m_toSelectModel.selectModel(data);
    

        if (m_localModel.numSubsets() > 1)
        {
            System.out.println("Not leaf");

            localInstances = m_localModel.split(data);
            data = null;
            m_sons = new ClassifierTree [m_localModel.numSubsets()];
            
            for (int i = 0; i < m_sons.length; i++)
            {

                m_sons[i] = getNewTree(localInstances[i],labels,false);
                localInstances[i] = null;
            }
        }
        else
        {
        	System.out.println("leaf");
            m_isLeaf = true;
            if (Utils.eq(data.sumOfWeights(), 0))
                m_isEmpty = true;
            data = null;
        }
    }
    
    
//****************** Get new tree
    
    protected ClassifierTree getNewTree(Instances data, int L, boolean status) throws Exception
    {
        ClassifierTree newTree = new ClassifierTree(m_toSelectModel,status);
        newTree.buildTree(data, false, L);
        return newTree;
    }
    
    /**
     * Classifies an instance.
     *
     * @param instance the instance to classify
     * @return the classification
     * @throws Exception if something goes wrong
     */

 public double  classifyInstance(Instance instance)throws Exception
    {
        double maxProb = -1;
        double currentProb;
        int maxIndex = 0;
        int j;
        
        int L=instance.classIndex();
        
        double []labels=new double[L];
        for(int i=0;i<L;i++)
        {

        for (j = 0; j < instance.numClasses(); j++)
        {
            currentProb = getProbs(i,j, instance, 1);
            if (Utils.gr(currentProb,maxProb))
            {
                maxIndex = j;
                maxProb = currentProb;
            }
        }
        }
        
        return (double)maxIndex;
    }
 
    public final double [] distributionForInstance(Instance instance,boolean useLaplace) throws Exception
    {
        int L = instance.classIndex();
        double [] labels = new double[L];
                for (int i = 0; i < L; i++)
                {
                    labels[i]=getProbsLaplace(i,1, instance, 1);
                    //System.out.print(labels[i]+" , ");
                }
                //System.out.print("\n");
        return labels;
    }
    
    /**
     * Help method for computing class probabilities of
     * a given instance.
     * @param classIndex the class index
     * @param instance the instance to compute the probabilities for
     * @param weight the weight to use
     * @return the probs
     * @throws Exception if something goes wrong
     */
  private double getProbs(int l,int classIndex, Instance instance, double weight) throws Exception
    {
        int L = instance.classIndex();
        
        double prob = 0;
        if (m_isLeaf)
        {
            return weight * m_localModel.classProb(l,classIndex, instance, -1);
        }
        else
        {
            int treeIndex = m_localModel.whichSubset(instance);
            if (treeIndex == -1)
            {
                double[] weights =m_localModel.weights(instance,l);
                for (int i = 0; i < m_sons.length; i++)
                {
                    if (!son(i).m_isEmpty)
                    {
                        prob += son(i).getProbs(l,classIndex, instance,weights[i] * weight);
                    }
                }
                return prob;
            }
            else
            {
                if (son(treeIndex).m_isEmpty)
                {
                    return weight * m_localModel.classProb(l,classIndex, instance,treeIndex);
                }
                else
                {
                    return son(treeIndex).getProbs(l,classIndex, instance, weight);
                }
            }
        }
    }
    
    private double getProbsLaplace(int l,int classIndex, Instance instance, double weight)
    throws Exception
    {
        
        int L = instance.classIndex();
        
        double prob = 0;
        if (m_isLeaf)
        {
        return weight * m_localModel.classProbLaplace(l,classIndex, instance, -1);
        }
        else
        {
            int treeIndex = m_localModel.whichSubset(instance);
            if (treeIndex == -1)
            {
                double[] weights = m_localModel.weights(instance,l);
                for (int i = 0; i < m_sons.length; i++)
                {
                    if (!son(i).m_isEmpty)
                    {
                        prob += son(i).getProbsLaplace(l,classIndex, instance,
                                                       weights[i] * weight);
                    }
                }
                return prob;
            }
            else
            {
                if (son(treeIndex).m_isEmpty)
                {
                    return weight * m_localModel.classProbLaplace(l,classIndex, instance,
                                                                  treeIndex);
                }
                else
                {
                    return son(treeIndex).getProbsLaplace(l,classIndex, instance, weight);
                }
            }
        }
    }
    /**
     * Cleanup in order to save memory.
     * 
     * @param justHeaderInfo
     */
    public final void cleanup(Instances justHeaderInfo) 
    {
      m_train = justHeaderInfo;
      m_test = null;
      if (!m_isLeaf)
        for (int i = 0; i < m_sons.length; i++)
  	m_sons[i].cleanup(justHeaderInfo);
    }
    /**
     * Method just exists to make program easier to read.
     */
    private ClassifierTree son(int index)
    {
        return (ClassifierTree)m_sons[index];
    }
 }
