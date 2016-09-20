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
 *    BinC45ModelSelection.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *
 */
 /*
 *    Extended by Reem Alotaibi to handle multi-label data, 2016
 *	  ra12404@bristol.a.cuk
 */
 

import java.util.Enumeration;
import weka.core.*;
import meka.core.MLUtils;

/**
 * Class for selecting a C4.5-like binary (!) split for a given dataset.
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision: 8034 $
 */
public class BinC45ModelSelection 
{
  /** for serialization */
  private static final long serialVersionUID = 179170923545122001L;

  /** Minimum number of instances in interval. */
  private int m_minNoObj;                     

  /** The FULL training dataset. */
  private Instances m_allData;
    
  /**
   * Initializes the split selection method with the given parameters.
   *
   * @param minNoObj minimum number of instances that have to occur in
   * at least two subsets induced by split
   * @param allData FULL training dataset (necessary for selection of
   * split points).  
   * @param useMDLcorrection whether to use MDL adjustement when
   * finding splits on numeric attributes
   */
  public BinC45ModelSelection(int minNoObj,Instances allData)
  {
    m_minNoObj = minNoObj;
    m_allData = allData;
  }

  /**
   * Sets reference to training data to null.
   */
  public void cleanup()
  {
    m_allData = null;
  }

  /**
   * Selects C4.5-type split for the given dataset.
   */
  public final ClassifierSplitModel selectModel(Instances data)
    {
    double averageInfoGain = 0;
    BinC45Split [] currentModel;
    BinC45Split bestModel = null;
    NoSplit noSplitModel =new NoSplit();
    double minQaluity;
    int validModels = 0;
    boolean multiVal = true;
    Distribution checkDistribution;
    double sumOfWeights;
        int i;
    
    try
        {
        int L = data.classIndex();
        // Check if all Instances belong to the same class or not enough instances to split on
        int []l=new int [L];
        int sum=0;
        for(i=0;i<L;i++)
        {
            checkDistribution = new Distribution(data,i);
            noSplitModel.addDistrbution(checkDistribution,i);
            if (Utils.sm(checkDistribution.total(),2*m_minNoObj)||Utils.eq(checkDistribution.total(),checkDistribution.perClass(checkDistribution.maxClass())))
            	l[i]=1;
            sum+=l[i];
        }
        if (sum==L)
            return noSplitModel;

        if (data.sumOfWeights()==0)
            return noSplitModel;

        // Check if all attributes are nominal and have a
        // lot of values.
        Enumeration enu = data.enumerateAttributes();
        while (enu.hasMoreElements())
        {
            Attribute attribute = (Attribute) enu.nextElement();
            if ((attribute.isNumeric()) ||
                (Utils.sm((double)attribute.numValues(),
                          (0.3*(double)m_allData.numInstances()))))
            {
                multiVal = false;
                break;
            }
        }
      currentModel = new BinC45Split[data.numAttributes()];
      sumOfWeights = data.sumOfWeights();
      
      // For each attribute.
      for (i = L; i < data.numAttributes(); i++)
      {
          // Get models for current attribute.
          if (i != 0)
          {
          currentModel[i] = new BinC45Split(i,m_minNoObj,sumOfWeights);
          currentModel[i].buildClassifier(data);
          
          // Check if useful split for current attribute
          // exists and check for enumerated attributes with 
          // a lot of values.
          if (currentModel[i].checkModel())
              if ((data.attribute(i).isNumeric()) ||
              (multiVal || Utils.sm((double)data.attribute(i).numValues(),(0.3*(double)m_allData.numInstances()))))
              {
                validModels++;
              }
       }
        else
        currentModel[i] = null;
      }
      // Check if any useful split was found.
      if (validModels == 0)
         return noSplitModel;
 
      averageInfoGain = averageInfoGain/(double)validModels;
      // Find "best" attribute to split on.
      minQaluity=0;
        
      for (i=L;i < data.numAttributes();i++)
      {
        if ((i != 0)&& (currentModel[i].checkModel()))
        {   
              if ( ( ( currentModel[i].infoGain() >= (averageInfoGain-1E-3) ) && ( Utils.gr(currentModel[i].gainRatio(),minQaluity)))||i==L)
              {
                  bestModel = currentModel[i];
                  minQaluity = currentModel[i].gainRatio();
                  bestModel.bestatt=bestModel.attIndex();
                  //Set the split point analogue to C45 if attribute numeric.
              }
          }
      }

     if (Utils.eq(minQaluity,0))
         return noSplitModel;
      // Add all Instances with unknown values for the corresponding
      // attribute to the distribution for the model, so that
      // the complete distribution is stored with the model. 
      //bestModel.distribution().addInstWithUnknown(data,bestModel.attIndex());

      return bestModel;
    }
      catch(Exception e)
    {
      e.printStackTrace();
    }
    return null;
  }

  /**
   * Selects C4.5-type split for the given dataset.
   */
  public final ClassifierSplitModel selectModel(Instances train, Instances test)
  {
    return selectModel(train);
  }
  
  /**
   * Returns the revision string.
   * 
   * @return		the revision
   */
  public String getRevision() {
    return RevisionUtils.extract("$Revision: 8034 $");
  }
}
