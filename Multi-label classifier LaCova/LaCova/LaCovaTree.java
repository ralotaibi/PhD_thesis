package meka.classifiers.multilabel.LaCova;
import meka.core.MLUtils;
import meka.core.PSUtils;
import meka.core.SuperLabelUtils;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Capabilities;
import weka.core.CapabilitiesHandler;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

/**
 * Class for handling a multi-label tree used for classification based on J48 algorithm.
 * @author Reem Al-Otaibi (ra12404@bristol.ac.uk)
 * @version May:2015 
 */

public class LaCovaTree implements  CapabilitiesHandler
{

  /** The model selection method. */  
  protected BinC45ModelSelection m_toSelectModel;     

  /** Local model at node. */
  protected ClassifierSplitModel m_localModel;  

  /** References to sons. */
  protected LaCovaTree [] m_sons;           

  /** True if node is leaf. */
  protected boolean m_isLeaf;
    
  /** True if node is single-label leaf. */
  protected boolean m_isSingleLabel;
  
  /** True if node is LP-label leaf. */
  protected boolean m_isLPLabel;

  /** True if node is empty. */
  protected boolean m_isEmpty;                  

  /** The training instances. */
  protected Instances m_train;                  

  /** The pruning instances. */
  protected Distribution m_test; 
  
  /** Number of label in the original space. */
  protected int org;

  /** The id for the node. */
  protected int m_id;
    
  /** The minimum number of instances at leaf. */
  protected int n;
    
  /** The variance. */
  protected double var;
    
  /** The covariance. */
  protected double cov;
    
  /** Stopping value for the covariance. */
  protected double thr_cov;
  
  /** Baseline for binary split. Note that any base classifier can be used instead of the J48*/
  protected J48 baseClassifier_J48=new J48();
      
  protected Classifier m_MultiClassifiers[] =null;
  
  protected Instances m_InstancesTemplates[] = null;
  
  protected int kMap[][] = null;

  protected CovarianceMatrix CM;
  
  protected static  int count=0;
  
  protected class pairCLass
  {
	  int[] pair;
	  double cov;
	  
	  public double getCov()
	  {
		  return cov;
	  }
	  public int[] getPair()
	  {
		  return pair;
	  }
  }
    
  /**
   * Constructor. 
   */
    public LaCovaTree(BinC45ModelSelection toSelectLocModel)
    {
     m_toSelectModel = toSelectLocModel;
    }
    
    public LaCovaTree(BinC45ModelSelection toSelectLocModel,boolean status)
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
  public void buildClassifier(Instances data, int original_labels) throws Exception
  {
	org=original_labels;
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
        baseClassifier_J48.setBinarySplits(true);
	    baseClassifier_J48.setCollapseTree(false);
	    baseClassifier_J48.setUnpruned(true);
	    baseClassifier_J48.setUseLaplace(true);
	
    	int L = data.classIndex();
    	m_test = null;
    	m_isLeaf = false;
    	m_isEmpty = false;
    	m_sons = null;
    	
    	CM=new CovarianceMatrix();

    	if (L==0)
    	{
    		var=CM.find_variance(data,1);
    		cov=CM.find_covariance(data,1);
    	}
    	else
    	{
    		var=CM.find_variance(data,L);
    		cov=CM.find_covariance(data,L);
    	}

		//Find best split model
    	m_localModel = m_toSelectModel.selectModel(data);
    	
		//Find the covariance threshold
		thr_cov=CM.find_thr_covariance(data,L);		

    	if((var==0) || ( m_localModel.m_numSubsets<=1))
    	{

    		System.out.println("Variance is low,,, Leaf!!");
    		m_isLeaf = true;
    		if (Utils.eq(data.sumOfWeights(), 0))
    			m_isEmpty = true;
    			data = null;
    	}
    	else if(cov<=thr_cov)
    	{
			System.out.println("Covariance is low,, Growing a Single-label Decision Tree!!");	
			
			//Using J48 as a base classifier
			m_MultiClassifiers = AbstractClassifier.makeCopies(baseClassifier_J48,L);
			m_InstancesTemplates = new Instances[labels];
            
			for(int j = 0; j < L; j++)
				{
				//count++;
					System.out.println("Single Binary Classifier");
					//Select only label attribute 'j'
					Instances D_j = MLUtils.keepAttributesAt(new Instances(data),new int[]{j},L);
					D_j.setClassIndex(0);
					//Build the classifier for each label
					m_MultiClassifiers[j].buildClassifier(D_j);
					m_InstancesTemplates[j] = new Instances(D_j, 0);
				}
			m_isLeaf = true;
			m_isSingleLabel=true;
			data=null;  
    	}
    	else
    	{   
    		System.out.println("Continue Growing Multi-label Tree!!");    
    		System.out.println("Split based on:"+m_localModel.bestatt);
            	localInstances = m_localModel.split(data);
            	data = null;
            	m_sons = new LaCovaTree [m_localModel.numSubsets()];
            	
            	for (int i = 0; i < m_sons.length; i++)
            	{    
            		m_sons[i] = getNewTree(localInstances[i],labels,false);
            		localInstances[i] = null;
          	    }   		   		
    }
}
    
  /**
   * Get a new Tree.
   *
   * @param instance the instance 
   * @param L is number of labels 
   * @param status
   * @return the tree
   * @throws Exception if something goes wrong
   */
    protected LaCovaTree getNewTree(Instances data, int L, boolean status) throws Exception
    {
        LaCovaTree newTree = new LaCovaTree(m_toSelectModel,status);
        newTree.buildTree(data, false, L);
        return newTree;
    }
    
    private int[] mapBack(Instances template, int i) 
    {
		try 
		{
			return MLUtils.toIntArray(template.classAttribute().value(i));
		} 
		catch(Exception e) 
		{
			return new int[]{};
		}
	}
    /**
     * Classifies an instance.
     *
     * @param instance the instance to classify
     * @return the classification
     * @throws Exception if something goes wrong
     */

 public int [] classifyInstance(Instance instance)throws Exception
    {
        double []labels=new double[instance.classIndex()];
        double []currentProb=new double[instance.classIndex()];
        double []maxProb =new double[instance.classIndex()];
        int [] maxIndex=new int[instance.classIndex()];
        int j;
                
        for (j = 0; j < instance.numClasses(); j++)
        {
            labels= getProbs(j, instance, 1);
            
            for(int i=0;i<labels.length;i++)
            {
            	currentProb[i]=labels[i];
            	if (Utils.gr(currentProb[i],maxProb[i]))
            	{
            		maxIndex[i] = j;
            		maxProb[i] = currentProb[i];
            	}
            }
        }
        
        return maxIndex;
    }
 
    public final double [] distributionForInstance(Instance instance,boolean useLaplace) throws Exception
    {
        int L = instance.classIndex();

        double [] labels = new double[L];

                	 if (!useLaplace) 
                		 labels = getProbs(1, instance, 1);
                	 else 
                		 labels = getProbsLaplace(1, instance, 1);   
                	 
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
  private double[] getProbs(int classIndex, Instance instance, double weight) throws Exception
    {
	  int L = instance.classIndex();
      double [] labels = new double[L];

      double []score=new double[1];
      double prob=0;
      if (m_isLeaf)
      {
    	  if (m_isSingleLabel==true)
          {  
    		  for(int m = 0; m < L; m++) 
      			{
    			  Instance x_j = (Instance)instance.copy();
    			  x_j.setDataset(null);
    			  x_j = MLUtils.keepAttributesAt(x_j,new int[]{m},L);
    			  x_j.setDataset(m_InstancesTemplates[m]);
    			  labels[m]=m_MultiClassifiers[m].distributionForInstance(x_j)[1];
      			}
              return labels;
          }
          else
          {
          	for(int j=0;j<L;j++)
          		labels[j]=weight * m_localModel.classProb(j,classIndex, instance, -1);
              return labels;
          }
      }
      else
      {
          int treeIndex = m_localModel.whichSubset(instance);
          if (treeIndex == -1)
          {
          	for(int j=0;j<L;j++)
          	{
          		prob=0;
          		double[] weights = m_localModel.weights(instance,j);  
          		
          		for (int i = 0; i < m_sons.length; i++)
          		{
          			if (!son(i).m_isEmpty)
          			{
                      score = son(i).getProbs(classIndex, instance,weights[i] * weight);
                      prob+=score[0];
          			}
          		}
              labels[j]=prob;
          	}
              return labels;
          }
          else
          {
              if (son(treeIndex).m_isEmpty)
              {
              	for(int j=0;j<L;j++)
              	{
              		labels[j]=weight * m_localModel.classProb(j,classIndex, instance,treeIndex);
              	}
                  return labels;
              }
              else
              {               	
                  return son(treeIndex).getProbs(classIndex, instance, weight);
              }
          }      
    }
}
    
    private double [] getProbsLaplace(int classIndex, Instance instance, double weight)
    throws Exception
    {       
        int L = instance.classIndex();
        double [] labels = new double[L];
        double []score=new double[1];
        double prob=0;
        if (m_isLeaf)
        {
      	  if (m_isSingleLabel==true)
            {  
      		  for(int m = 0; m < L; m++) 
        			{
      			  Instance x_j = (Instance)instance.copy();
      			  x_j.setDataset(null);
      			  x_j = MLUtils.keepAttributesAt(x_j,new int[]{m},L);
      			  x_j.setDataset(m_InstancesTemplates[m]);
      			  labels[m]=m_MultiClassifiers[m].distributionForInstance(x_j)[1];
        			}
                return labels;
            }
            else
            {
            	for(int j=0;j<L;j++)
            		labels[j]=weight * m_localModel.classProbLaplace(j,classIndex, instance, -1);
                return labels;
            }
        }
        else
        {
            int treeIndex = m_localModel.whichSubset(instance);
            if (treeIndex == -1)
            {
            	for(int j=0;j<L;j++)
            	{
            		prob=0;
            		double[] weights = m_localModel.weights(instance,j);  
            		
            		for (int i = 0; i < m_sons.length; i++)
            		{
            			if (!son(i).m_isEmpty)
            			{
                        score = son(i).getProbsLaplace(classIndex, instance,weights[i] * weight);
                        prob+=score[0];
            			}
            		}
                labels[j]=prob;
            	}
                return labels;
            }
            else
            {
                if (son(treeIndex).m_isEmpty)
                {
                	for(int j=0;j<L;j++)
                	{
                		labels[j]=weight * m_localModel.classProbLaplace(j,classIndex, instance,treeIndex);
                	}
                    return labels;
                }
                else
                {               	
                    return son(treeIndex).getProbsLaplace(classIndex, instance, weight);
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
    private LaCovaTree son(int index)
    {
        return (LaCovaTree)m_sons[index];
    }
    
 }
