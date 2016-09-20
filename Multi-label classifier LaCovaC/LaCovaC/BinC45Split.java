package meka.classifiers.multilabel.LaCovaC;

import weka.core.Instances;
import weka.core.Instance;
import weka.core.Utils;
import weka.core.RevisionUtils;

import java.util.*;

import meka.classifiers.multilabel.LaCovaC.ClassifierSplitModel;
import meka.classifiers.multilabel.LaCovaC.CovarianceMatrix;
import meka.classifiers.multilabel.LaCovaC.Distribution;
import meka.classifiers.multilabel.LaCovaC.ExtendMatrix;
import weka.core.matrix.*;

 /*
 *    Extended by Reem Alotaibi to handle multi-label data, 2016
 *	  ra12404@bristol.a.cuk
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

  /** Variance of split. */
  private double [] Qvar;

  /** Covariance of split.  */
  private double [] Qcov;

  /** Quality of the split. */
  private double quality;

  /**
   * Initializes the split model.
   */
  public BinC45Split(int attIndex,int minNoObj,double sumOfWeights)
  {
    // Get index of attribute to split on.
    m_attIndex = attIndex;
        
    // Set minimum number of objects.
    m_minNoObj = minNoObj;
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
    quality = 0;
    var=0;
    cov=0;
     
    // Different treatment for enumerated and numeric attributes.
    if (trainInstances.attribute(m_attIndex).isNominal())
    {
      handleEnumeratedAttribute(trainInstances);
    }
    else
    {
      trainInstances.sort(trainInstances.attribute(m_attIndex));
      handleNumericAttribute(trainInstances);
      //handleNumericAttributeFast(trainInstances);  //This method applied an incremental update for the covraince matrix, which should be faster
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
  
  /**
   * Returns Quality for the generated split.
   */
    public final double Quality()
    {
     return quality;
    }
    /**
     * Returns variance for the generated split.
     */
    public  double  variance()
    {
        int i;
        double sum=0;
        for (i=0; i < Qvar.length; i++)
            sum+=Qvar[i];
       
        return sum;
    }
    /**
     * Returns covariance for the generated split.
     */
    public double  covarinace()
    {
        int i;
        double sum=0;
        for (i=0; i < Qcov.length; i++)
            sum+=Qcov[i];
        
        return sum;
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
        CovarianceMatrix CM=new CovarianceMatrix();
        double []Q;
        double []n;
        double minQS;
        int S0=0,S1=0;
        List<Double> thisLabel0=new ArrayList<Double>();
        List<Double> thisLabel1=new ArrayList<Double>();
        List<List<Double>> labelsListS0 = new ArrayList<List<Double>>();
        List<List<Double>> labelsListS1 = new ArrayList<List<Double>>();
        
        int numAttValues=0;
        numAttValues = trainInstances.numDistinctValues(m_attIndex);
        Qvar=new double[numAttValues];
        Qcov=new double[numAttValues];
        Q=new double[numAttValues];
        n=new double[numAttValues];
        minQS=Double.MAX_VALUE;
        
        for(int i=0;i<numAttValues;i++)
        {
            Qvar[i]=Double.MAX_VALUE;
            Qcov[i]=Double.MAX_VALUE;
        }
            
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
            bestDisMap.put(j,d);
        }
        
        if(S0>1 && S1>1)
            m_numSubsets=2;
                
        //Generate Split for the attribute
        Instances [] instancesSplit;
        instancesSplit=split_instances(trainInstances,m_attIndex,numAttValues);

        //Calculate Covariance Matrix
        for (int i=0;i<instancesSplit.length;i++)
        {
            n[i]=instancesSplit[i].numInstances();
           if (n[i]>0)
           {
            Qvar[i]=CM.find_variance(instancesSplit[i],L);
            Qcov[i]=CM.find_covariance(instancesSplit[i],L);
           // System.out.println("var["+i+"]="+Qvar[i]);
            //System.out.println("cov["+i+"]="+Qcov[i]);
            if(L==1)
            {
            	//System.out.println("Single label");
            	 Q[i]=Qvar[i];	
            }
            else
            {
            	Q[i]=Math.min(Qvar[i],Qcov[i]);
            }
           }
            if ((i == 0) || Q[i]<minQS)
            {
                minQS = Q[i];
                m_splitPoint = (double)i;
            }
        }
        //quality=(Qvar[0]*n[0]+Qvar[1]*n[1])/N+(Qcov[0]*n[0]+Qcov[1]*n[1])/N;
        //quality=Math.min((Qvar[0]*n[0]+Qvar[1]*n[1])/N,(Qcov[0]*n[0]+Qcov[1]*n[1])/N);

        for (int i=0;i<instancesSplit.length;i++)
        {
        quality+=(Q[i]*n[i])/N;
        }
    }
    //Method to Split instances into two bins
    public final Instances [] split_instances(Instances data, int index, int num) throws Exception
    {
        
        Instances [] instances = new Instances [num];
        Instance instance;
        int  i, j;
        
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
        int numAttValues;
        numAttValues = trainInstances.numDistinctValues(m_attIndex);
        int N=trainInstances.numInstances();

        CovarianceMatrix CM=new CovarianceMatrix();
        double []currVar=new double[numAttValues];
        double []currCov=new double[numAttValues];
        double []currMin=new double[numAttValues];
        double currminQS=Double.MAX_VALUE;
        
        Qvar=new double[numAttValues];
        Qcov=new double[numAttValues];
        double minQS=Double.MAX_VALUE;
        double Q[],n[],q = 0;
        Q=new double[numAttValues];
        n=new double[numAttValues];
        
        // 2 to split the numerical attribute into two bags, 2 for binary labels, can be more in a multi-target
        Distribution d=new Distribution(2,2);
                
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
                    
                    for (int i=0;i<InstancesBags.length;i++)
                    {
                        n[i]=InstancesBags[i].numInstances();
                        if (n[i]>0)
                        {
                        currVar[i]=CM.find_variance(InstancesBags[i],L);
                        currCov[i]=CM.find_covariance(InstancesBags[i],L);
                        if(L==1)
                        {
                        	//System.out.println("Single label");
                        	currMin[i]=Qvar[i];	
                        }
                        else
                        {
                        currMin[i]=Math.min(currVar[i],currCov[i]);
                        }
                        currMin[i]=currMin[i]*n[i];
                        }
                        q+=currMin[i];
                        total+=n[i];
                    }
                    q=q/total;
                    if (Utils.sm(q,currminQS))
                    {
                      currminQS = q;
                      splitIndex = next-1;
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
    
    // Set instance variables' values to values for best split.
    m_numSubsets = 2;
    m_splitPoint =(trainInstances.instance(splitIndex+1).value(m_attIndex)+trainInstances.instance(splitIndex).value(m_attIndex))/2;

    // In case we have a numerical precision problem we need to choose the smaller value
    if (m_splitPoint == trainInstances.instance(splitIndex + 1).value(m_attIndex))
    {
    m_splitPoint = trainInstances.instance(splitIndex).value(m_attIndex);
    }
    //System.out.println("split="+m_splitPoint);

    // Restore distributioN for best split.
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

    //Calculate the Covariance Matrix for the best split
        total=0;
    for (int i=0;i<instancesSplit.length;i++)
    {
        n[i]=instancesSplit[i].numInstances();
        if (n[i]>0)
        {
            Qvar[i]=CM.find_variance(instancesSplit[i],L);
            Qcov[i]=CM.find_covariance(instancesSplit[i],L);
            if(L==1)
            {
            	//System.out.println("Single label");
            	Q[i]=Qvar[i];	
            }
            else
            {
                Q[i]=Math.min(Qvar[i],Qcov[i]);
            }
 
        }
        if ((i == 0) || Q[i]<minQS)
        {
            minQS = Q[i];
        }
        total+=n[i];
    }
    for (int i=0;i<instancesSplit.length;i++)
    {
    quality+=(Q[i]*n[i])/N;
    }
}
    /**
     * Creates split on numeric attribute.
     *
     * @exception Exception if something goes wrong
     */
    private void handleNumericAttributeFast(Instances trainInstances)throws Exception
    {
        int next = 1;
        int last = 0;
        int index = 0;
        int splitIndex = -1;
        double minSplit;
        
        int L = trainInstances.classIndex();
        int N=trainInstances.numInstances();
        int numAttValues;
        numAttValues = trainInstances.numDistinctValues(m_attIndex);
        
        CovarianceMatrix CM=new CovarianceMatrix();
        double []currVar=new double[numAttValues];
        double []currCov=new double[numAttValues];
        double []currMin=new double[numAttValues];
        double bestQS=Double.MAX_VALUE;
        
        Qvar=new double[numAttValues];
        Qcov=new double[numAttValues];
        double Q;

        // 2 to split the numerical attribute into two bags, 2 for binary labels, can be more in a multi-target
        Distribution d=new Distribution(2,2);
        
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
        
        
        //intiliaze the cij of all the training instances, M1 is already zeros
        double [][]m=new double[L][L];
        double [][]m1=new double[L][L];
        double [][]m2=new double[L][L];
        double n1=0,n2=N;
        m2=CM.find_cij(trainInstances,L);
        //Define Matrices
        ExtendMatrix EC=new ExtendMatrix();
        Matrix M=new Matrix(L,L);
        Matrix M1=new Matrix(L,L);
        Matrix M2=new Matrix(L,L);
        M1.constructWithCopy(m1);
        M2.constructWithCopy(m2);

        //we start by putting all instances in one bag and then move each two instances to check the split cut.
        while (next < N)
        {
            
            if (trainInstances.instance(next-1).value(m_attIndex)+1e-5 <trainInstances.instance(next).value(m_attIndex))
            {
                
                // Move all label-class values for all Instances up to next possible split point.
               
                for(int i = last; i < next; i++)
                {
                    InstancesBags[0].add(trainInstances.instance(i));
                }
                //Find cij for the instances in this cut-point
                m=CM.find_cij(InstancesBags[0],L);
                M.constructWithCopy(m);
                
                //Update the cij matrices
                M1.plusEquals(M); n1=n1+(next-last);
                M2.minusEquals(M); n2=n2-(next-last);
                
                //Define Matrices diagonal
                SingularValueDecomposition M1S=new SingularValueDecomposition(M1);
                SingularValueDecomposition M2S=new SingularValueDecomposition(M2);

                // Check if enough Instances in each subset and compute values for criteria.
                if (Utils.grOrEq(n1,minSplit) && Utils.grOrEq(n2,minSplit))
                {
                    //For the first split
                    currVar[0]=M1.trace();
                    //currCov[0]=(1/(Math.pow(n1,2))*Math.sum(Math.abs(n1*M1-diag(M1)*diag(M1)T));
                    
                   //n1*M1
                    Matrix M1n=M1.times(n1);
                    
                   //diag(M1)*diag(M1)T
                    double []diag1=M1S.getSingularValues();
                    Matrix mdiag1=EC.Vector_to_Matrix(diag1,L,1);
                    Matrix t_mdiag1=mdiag1.transpose();
                    Matrix mul=mdiag1.times(t_mdiag1);
                    
                    //n1*M1-(diag(M1)*diag(M1)T)
                    Matrix min=M1n.minus(mul);
                    
                    //sum|min|
                    double sum=EC.sum_abs(min);
                    currCov[0]=1/(Math.pow(n1,2))*sum;
                    currMin[0]=Math.min(currVar[0],currCov[0]);
                       
                    //For the second split
                    currVar[1]=M2.trace();
                    //currCov[1]=(1/(Math.pow(n2,2))*Math.sum(Math.abs(n2*M2-diag(M2)*diag(M2)T));
                    //n1*M1
                    Matrix M2n=M2.times(n2);
                    //diag(M1)*diag(M1)T
                    double []diag2=M2S.getSingularValues();
                    Matrix mdiag2=EC.Vector_to_Matrix(diag2,L,1);
                    Matrix t_mdiag2=mdiag2.transpose();
                    Matrix mul2=mdiag2.times(t_mdiag2);
                    //n2*M2-(diag(M2)*diag(M2)T)
                    Matrix min2=M2n.minus(mul2);
                    //sum|min|
                    sum=EC.sum_abs(min2);
                    currCov[1]=1/(Math.pow(n2,2))*sum;
                    currMin[1]=Math.min(currVar[1],currCov[1]);

                    Q=(currMin[0]*n1)/N+(currMin[1]*n2)/N;
                    
                    if (Utils.sm(Q,bestQS))
                    {
                        bestQS = Q;
                        splitIndex = next-1;
                        Qcov[0]=currCov[0];
                        Qcov[1]=currCov[1];
                        Qvar[0]= currVar[0];
                        Qvar[1]= currVar[1];
                        quality=currMin[0]+currMin[1];
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
        
        // Set instance variables' values to values for best split.
        m_numSubsets = 2;
        m_splitPoint =(trainInstances.instance(splitIndex+1).value(m_attIndex)+trainInstances.instance(splitIndex).value(m_attIndex))/2;
        
        // In case we have a numerical precision problem we need to choose the smaller value
        if (m_splitPoint == trainInstances.instance(splitIndex + 1).value(m_attIndex))
        {
            m_splitPoint = trainInstances.instance(splitIndex).value(m_attIndex);
        }
        
        // Restore distributioN for best split.
        for(int j=0; j<L; j++)
        {
            d.addRange(0,trainInstances,0,splitIndex+1,j);
            d.addRange(1,trainInstances,splitIndex+1,N,j);
            bestDisMap.put(j,d);
        }
                                  
    quality=quality/N;
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
  @Override
  /**
   * Returns a string containing java source code equivalent to the test made at
   * this node. The instance being tested is called "i".
   * 
   * @param index index of the nominal value tested
   * @param data the data containing instance structure info
   * @return a value of type 'String'
   */
  public final String sourceExpression(int index, Instances data) 
  {System.out.println("F1");
    StringBuffer expr = null;
    if (index < 0) 
    {
      return "i[" + m_attIndex + "] == null";
    }
    if (data.attribute(m_attIndex).isNominal()) 
    {
      if (index == 0) 
      {
        expr = new StringBuffer("i[");
      } 
      else 
      {
        expr = new StringBuffer("!i[");
      }
      expr.append(m_attIndex).append("]");
      expr.append(".equals(\"")
        .append(data.attribute(m_attIndex).value((int) m_splitPoint))
        .append("\")");
    } 
    else 
    {
      expr = new StringBuffer("((Double) i[");
      expr.append(m_attIndex).append("])");
      if (index == 0) 
      {
        expr.append(".doubleValue() <= ").append(m_splitPoint);
      } 
      else 
      {
        expr.append(".doubleValue() > ").append(m_splitPoint);
      }
    }
    return expr.toString();
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
