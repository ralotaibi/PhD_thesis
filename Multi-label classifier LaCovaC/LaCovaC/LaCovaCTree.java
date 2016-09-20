package meka.classifiers.multilabel.LaCovaC;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;

import com.sun.xml.internal.ws.policy.privateutil.PolicyUtils.Text;

import meka.core.A;
import meka.core.MLUtils;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.CapabilitiesHandler;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import meka.classifiers.multilabel.LaCovaC.LabelList;
import meka.classifiers.multilabel.LaCovaC.CorrelationMatrix;
import meka.classifiers.multilabel.LaCovaC.CovarianceMatrix;
import meka.classifiers.multilabel.LaCovaC.LaCovaCTree;
import meka.classifiers.multilabel.LaCovaC.ClusteringCorr;

/**
 * Class for handling a multi-label tree used for classification based on J48 algorithm.
 * @author Reem Al-Otaibi (ra12404@bristol.ac.uk)
 * @version August:2016 
 */

public class LaCovaCTree implements  CapabilitiesHandler
{

  /** The model selection method. */  
  protected BinC45ModelSelection m_toSelectModel;     

  /** Local model at node. */
  protected ClassifierSplitModel m_localModel;  

  /** References to sons. */
  protected LaCovaCTree [] m_sons;           

  /** True if node is leaf. */
  protected boolean m_isLeaf;
  
  /** True if node is leaf. */
  protected int num_labels;
    
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
  
  protected int [][]cls;
    
  /** Stopping value for the covariance. */
  protected double thr_cov;
  
  /** Baseline for binary split. Note that any base classifier can be used instead of the J48*/
  protected J48 baseClassifier_J48=new J48();
      
  protected Classifier m_MultiClassifiers[] =null;
  
  protected Instances m_InstancesTemplates[] = null;
  
  protected int kMap[] = null;
  
  protected int LMap[] = null;

  protected CovarianceMatrix CM;
  
  protected CorrelationMatrix CR;
    
  protected ClusteringCorr clusters;
  
  protected int []indices;
  
  protected LabelList clus=new LabelList();
  
  protected static  List<Double> labels = new ArrayList<Double>();
  
  protected static  HashMap<Integer,Double> map = new HashMap<Integer,Double>();
  
  protected static  HashMap<Integer,Double> prediction = new HashMap<Integer,Double>();
  
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
    public LaCovaCTree(BinC45ModelSelection toSelectLocModel)
    {
    	m_toSelectModel = toSelectLocModel;
    	map = new HashMap<Integer,Double>();
     	labels = new ArrayList<Double>();
    }
    
    public LaCovaCTree(BinC45ModelSelection toSelectLocModel,boolean status)
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
    
    indices=new int[L];
    
    for(int i=0;i<L;i++)
    	indices[i]=i;
    
    buildTree(data, false,indices);
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
  public void buildTree(Instances data, boolean keepData, int[] inx) throws Exception
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
	    	
	    	CR=new CorrelationMatrix();
	    	CM=new CovarianceMatrix();

	    	if (L==0)
	    	{
	    		var=CM.find_variance(data,1);
	    		cov=CM.find_covariance(data,1);
	        	num_labels=data.classIndex()+1;
	    	}
	    	else
	    	{
	    		var=CM.find_variance(data,L);
	    		cov=CM.find_covariance(data,L);
	        	num_labels=data.classIndex();
	    	}
	    	    	
			//Clustering single-linkage   	
	    	clusters=new ClusteringCorr();
	        clus=clusters.generate_clusters(data,L,"single",inx);       

			//Find best split model
	    	m_localModel = m_toSelectModel.selectModel(data);
	    	
	    	if((var==0) || ( m_localModel.m_numSubsets<=1))
	    	{
	    		LP(data);   		
	    		//System.out.println("Leaf!!");
	    		//System.out.println(m_InstancesTemplates[0].toString());
	    		m_isLeaf = true;
	    		LMap=inx.clone();
	   
	    		if (Utils.eq(data.sumOfWeights(), 0))
	    			m_isEmpty = true;
	    			data = null;
	    	}
	    	else
	    	{   
	    		//LaCova-CLus    		
	    		//Split vertically to cluster labels if there is clusters
	    		 if(clus.indices.length>1)
	    		{ 
	     			 System.out.println("Vertical Split="+clus.indices.length);
	     			 m_InstancesTemplates=new Instances[clus.indices.length];
	    			 m_InstancesTemplates =clusters.convert(data,clus.indices);
		             m_sons = new LaCovaCTree [clus.indices.length];
		             	    		
	    			 for(int j = 0; j < clus.indices.length; j++)
	 					{
	    				 kMap=clus.names[j];
	    				 m_sons[j] = getNewTree(m_InstancesTemplates[j],false,clus.names[j].clone()); 
	 					}
	    			m_isLPLabel=true;
	    			data=null; 
	    		}
	    		else
	    		{ 
	    			System.out.println("Horizontal Split!!");                         	

	            	localInstances = m_localModel.split(data);
	            	data = null;
	            	m_sons = new LaCovaCTree [m_localModel.numSubsets()];
	            	
	            	for (int i = 0; i < m_sons.length; i++)
	            	{
	            		System.out.println("HS:"+(i+1));
	   				    kMap=inx.clone(); 
	            		m_sons[i] = getNewTree(localInstances[i],false,inx.clone());
	            		localInstances[i] = null;
	          	    }
	    		}  
	    	}
	}
  protected void LP (Instances data) throws Exception 
  {
//Create a nominal class attribute of all (existing) possible combinations of labels as possible values
	int C = data.classIndex();
	FastVector ClassValues = new FastVector(C);
	HashSet<String> UniqueValues = new HashSet<String>();
	for (int i = 0; i < data.numInstances(); i++) {
		UniqueValues.add(MLUtils.toBitString(data.instance(i),C));
	}
	Iterator<String> it = UniqueValues.iterator();
	while (it.hasNext()) {
		ClassValues.addElement(it.next());
	}
	Attribute NewClass = new Attribute("Class", ClassValues);

	//Filter Remove all class attributes
	Remove FilterRemove = new Remove();
	FilterRemove.setAttributeIndices("1-"+C);
	FilterRemove.setInputFormat(data);
	Instances NewTrain = Filter.useFilter(data, FilterRemove);

	//Insert new special attribute (which has all possible combinations of labels) 
	NewTrain.insertAttributeAt(NewClass, 0);
	NewTrain.setClassIndex(0);

	//Add class values
	for (int i = 0; i < NewTrain.numInstances(); i++) 
	{
		String comb = MLUtils.toBitString(data.instance(i),C);
		NewTrain.instance(i).setClassValue(comb);
	}

	// keep the header of new dataset for classification
	m_InstancesTemplates = new Instances[1];
	m_InstancesTemplates[0] = NewTrain;
	m_MultiClassifiers=	m_MultiClassifiers=AbstractClassifier.makeCopies(baseClassifier_J48,1);
	m_MultiClassifiers[0].buildClassifier(m_InstancesTemplates[0]);
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
    protected LaCovaCTree getNewTree(Instances data,  boolean status,int[] arr) throws Exception
    {
        LaCovaCTree newTree = new LaCovaCTree(m_toSelectModel,status);
        newTree.buildTree(data, false,arr.clone());
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
 
    public final double [] distributionForInstance(Instance instance, boolean useLaplace) throws Exception
    {
    	 int L = instance.classIndex();
         double [] labels = new double[L]; 
         double [] predictions = new double[L];      
         double [] arr=new double[L];

                 	 if (!useLaplace) 
                 		 predictions = getProbs(1, instance, 1);
                 	 else 
                 	     predictions = getProbsLaplace(1, instance, 1);//Laplace 
                 	                 	                     
        
        //("------CONFIDENCE------");       
        for(int i=0;i<map.size();i++)
        {
     	   arr[i] = map.get(i);
        }
        
        //("------PREDICTION------");
        for(int i=0;i<prediction.size();i++)
        {
     	   labels[i]=prediction.get(i);
        }

         return  labels; //arr return the score per label
    }
    private double [] getProbsLaplace(int classIndex, Instance instance, double weight)
    	    throws Exception
    	    {      
    	 int L = instance.classIndex();
	        double []score=new double[1];
	        double prob=0;
	        		 
	        	 if (m_isLeaf)
	             {
	        		 Instance x_ =(Instance) instance.copy(); 
	        		  x_.setDataset(null);
	        		  for (int i = 0; i < instance.classIndex(); i++)
	        			  x_.deleteAttributeAt(0);
	        		  x_.insertAttributeAt(0);
	        		  x_.setDataset(m_InstancesTemplates[0]);
	        		  
	        		  double result[] = new double[x_.numClasses()];

	        		  result[(int)m_MultiClassifiers[0].classifyInstance(x_)] = 1.0;   	        		  
	        		  double[]d=m_MultiClassifiers[0].distributionForInstance(x_);
	        		  int m=(int)m_MultiClassifiers[0].classifyInstance(x_);
	        		  String s=m_InstancesTemplates[0].classAttribute().value(m);   	        		
	        		  
	        		  double k_indices[] = MLUtils.fromBitString(s);
	        		 
	        		  for(int j=0;j<LMap.length;j++)
	        		  {
	        			 if(k_indices[j]==1)
	        				 map.put(LMap[j],  d[0])	;
	        			 else
	        				 map.put(LMap[j],  1-d[0])	;
	        		  }

	        		  //Make prediction
	        		  int c = instance.classIndex();
	        		  double y[] = new double[org];

	        		 // Get a meta classification
	        		  for(int j=0;j<LMap.length;j++)
	        		  {	
	        			  if(k_indices[j]==1)
	        				 prediction.put(LMap[j],  1.0)	;   	 
	        			  else
	        				  prediction.put(LMap[j],  0.0)	;   	        		
	        		  }
	        		  return y;

	             }    	        	 
	        	 else if(m_isLPLabel)
	        	 {
      	    	 //System.out.println("LP!");
	        		 double [] x_labels=null;
	        		 int j=0;
	        		 for(int [] aGroup:clus.names)
	        			{
	        			 x_labels=new double[aGroup.length];
	        			 Instance x_=MLUtils.setTemplate(instance,m_InstancesTemplates[j]);    	        		     
	        		     int whichClus = j;
	        		     x_labels=son(whichClus).getProbsLaplace(classIndex, x_, weight); 
        			     j++;
						}
	        		 return x_labels;
	        	 }
	        	 else
	             {
      	    	 double [] x_labels = new double[num_labels];
      	    	 
	                 int treeIndex = m_localModel.whichSubset(instance);
	                 if (treeIndex == -1)
	                 {
	                 	for(int j=0;j<num_labels;j++)
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
	                     x_labels[j]=prob;
	                 	}
	                     return x_labels;
	                 }//end if
	                 else
	                 {   
	                     if (son(treeIndex).m_isEmpty)
	                     { 
	                     	for(int j=0;j<num_labels;j++)
	                     	{
	                     		x_labels[j]=weight * m_localModel.classProbLaplace(j,classIndex, instance,treeIndex);
	                     		 map.put(kMap[j],  x_labels[j])	;
	     	        			 prediction.put(LMap[j],  x_labels[j])	; 
	                     	}
	                         return x_labels;
	                     }
	                     else
	                     {        	
	                         return son(treeIndex).getProbsLaplace(classIndex, instance, weight);
	                     }
	                 }//end else
	             }//end else
    	  }//end method
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
      double []score=new double[1];
      double prob=0;
      		 
      	 if (m_isLeaf)
           {
    		  
	         double [] x_labels = new double[num_labels];
      		 for(int j=0;j<num_labels;j++)
      		 {
      			 x_labels[j]=weight * m_localModel.classProb(j,classIndex, instance, -1);
               	 map.put(kMap[j],  x_labels[j])	;
      		 }    	        		    	        	        		 
               return x_labels;      	        		 
           }    	        	 
      	 else if(m_isLPLabel)
      	 {
   	    	 //System.out.println("LP!");
      		 double [] x_labels=null;
      		 int j=0;
      		 for(int [] aGroup:clus.names)
      			{
      			 x_labels=new double[aGroup.length];
      			 Instance x_=MLUtils.setTemplate(instance,m_InstancesTemplates[j]);    	        		     
	        		     int whichClus = j;
      		     x_labels=son(whichClus).getProbs(classIndex, x_, weight); 
     			     j++;
					}
      		 return x_labels;
      	 }
      	 else
           {
   	    	 double [] x_labels = new double[num_labels];
   	    	 
               int treeIndex = m_localModel.whichSubset(instance);
               if (treeIndex == -1)
               {
               	for(int j=0;j<num_labels;j++)
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
                   x_labels[j]=prob;
               	}
                   return x_labels;
               }//end if
               else
               {   
                   if (son(treeIndex).m_isEmpty)
                   { 
                   	for(int j=0;j<num_labels;j++)
                   	{
                   		x_labels[j]=weight * m_localModel.classProb(j,classIndex, instance,treeIndex);
                   		 map.put(kMap[j],  x_labels[j])	;
                   	}
                       return x_labels;
                   }
                   else
                   {        	
                       return son(treeIndex).getProbs(classIndex, instance, weight);
                   }
               }//end else
           }//end else
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
    private static long PRINTED_NODES = 0;

    protected static long nextID() 
    {
        return PRINTED_NODES++;
    }
    /**
     * Returns source code for the tree as an if-then statement. 
     * @param className the classname that this static classifier has
     * @return an array containing two stringbuffers, the first string containing
     *         assignment code, and the second containing source for support code.
     * @throws Exception if something goes wrong
     */
    public StringBuffer[] toSource(String className) throws Exception 
    {
        StringBuffer[] result = new StringBuffer[2];
        if (m_isLeaf) 
        {
        	for(int j=0;j<num_labels;j++)
        	{
        		result[0] = new StringBuffer("p["+0+"] = "+ m_localModel.bestDisMap.get(0).maxClass(0)+ ";\n");
        		result[1] = new StringBuffer("");
        		System.out.println(result.toString());
        	}
        } 
        else 
        {
        	System.out.println("ELSE");
          StringBuffer text = new StringBuffer();
          StringBuffer atEnd = new StringBuffer();

          long printID = LaCovaCTree.nextID();

          text.append("  static double N")
            .append(Integer.toHexString(m_localModel.hashCode()) + printID)
            .append("(Object []i) {\n").append("    double p = Double.NaN;\n");

          text.append("    if (")
            .append(m_localModel.sourceExpression(-1, m_train)).append(") {\n");
          System.out.println(text.toString());
          
          for(int j=0;j<num_labels;j++)
      		{
        	  text.append("p["+j+"] = "+ m_localModel.bestDisMap.get(j).maxClass(0)+ ";\n");
        	  System.out.println(text.toString());
      		}  
          
          text.append("    } ");
          System.out.println(text.toString());
          
          for (int i = 0; i < m_sons.length; i++) 
          {
            text.append("else if (" + m_localModel.sourceExpression(i, m_train)
              + ") {\n");
            System.out.println(text.toString());
            if (m_sons[i].m_isLeaf) 
            {
            	for(int j=0;j<num_labels;j++)
            		text.append("p["+j+"] = "+ m_localModel.bestDisMap.get(j).maxClass(0)+ ";\n");
            } 
            else 
            {
              StringBuffer[] sub = m_sons[i].toSource(className);
              text.append(sub[0]);
              atEnd.append(sub[1]);
            }
            text.append("    } ");
            if (i == m_sons.length - 1) 
            {
              text.append('\n');
            }
          }

          text.append("    return p;\n  }\n");
          result[0] = new StringBuffer("    p = " + className + ".N");
          result[0].append(Integer.toHexString(m_localModel.hashCode()) + printID)
            .append("(i);\n");
          result[1] = text.append(atEnd);
        }
        result.toString();
        return result;
      }
    /**
     * Method just exists to make program easier to read.
     */
    private LaCovaCTree son(int index)
    {
        return (LaCovaCTree)m_sons[index];
    }
    
 }
