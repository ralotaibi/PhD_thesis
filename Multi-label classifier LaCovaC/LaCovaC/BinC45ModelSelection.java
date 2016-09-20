package meka.classifiers.multilabel.LaCovaC;
import java.util.Enumeration;
import weka.core.*;

 /*
 *    Extended by Reem Alotaibi to handle multi-label data, 2016
 *	  ra12404@bristol.a.cuk
 */
public class BinC45ModelSelection 
{

  /** Minimum number of instances in interval. */
  private int m_minNoObj;                     

  /** The FULL training data set. */
  private Instances m_allData;
    
  /**
   * Initializes the split selection method with the given parameters.
   *
   * @param minNoObj minimum number of instances that have to occur in
   * at least two subsets induced by split
   * @param allData FULL training data set (necessary for selection of
   * split points).  
   * @param useMDLcorrection whether to use MDL adjustment when
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
   * Selects C4.5-type split for the given data set.
   */
  public final ClassifierSplitModel selectModel(Instances data)
    {
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
          //
         // System.out.println("Attribute="+(i+1)+" Cov="+currentModel[i].covarinace()+" Var="+currentModel[i].variance()
        		 // +" Qulaity="+currentModel[i].Quality());
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

      // Find "best" attribute to split on.
      minQaluity=Double.MAX_VALUE;
        
      for (i=L;i < data.numAttributes();i++)
      {
        if ((i != 0)&& (currentModel[i].checkModel()))
          {
              if ((currentModel[i].Quality() < minQaluity)||(i==L))
              {
                  bestModel = currentModel[i];
                  minQaluity = currentModel[i].Quality();
                  bestModel.var=bestModel.variance();
                  bestModel.cov=bestModel.covarinace();
                  bestModel.bestatt=bestModel.attIndex();
                  bestModel.thr=bestModel.splitPoint();
                  // Set the split point analog to C45 if attribute numeric.
              }
          }
      }
     //System.out.println("Best attribute:"+(bestModel.bestatt)+" threshold="+ bestModel.splitPoint());

     if (Utils.eq(minQaluity,Double.MAX_VALUE))
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
   * Selects C4.5-type split for the given data set.
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
  public String getRevision() 
  {
    return RevisionUtils.extract("$Revision: 8034 $");
  }
}
