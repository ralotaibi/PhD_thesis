
/**
 * Implementation of LaCova: a decision tree classifier for multi-label based on J48 algorithm. 
 * Al-Otaibi, RM, Kull, M & Flach, PA 2014, ‘LaCova: A Tree-Based Multi-Label Classifier using Label Covariance 
 * as Splitting Criterion’. in: International Conference on Machine Learning and Applications (ICMLA). 
 * IEEE Computer Society, Detroit, MI, pp. 74
 * @author Reem Al-Otaibi (ra12404@bristol.ac.uk)
 * @version May:2015 
 */
package meka.classifiers.multilabel.LaCova;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.Random;

import meka.core.A;
import meka.core.F;
import meka.core.MLEvalUtils;
import meka.core.MLUtils;
import meka.core.Result;
import weka.core.*;
import weka.core.converters.ConverterUtils.DataSource;
import meka.classifiers.multilabel.LaCova.*;

 @SuppressWarnings("serial")
public class FeatureTest 
{
  /** for serialization */ 
  private static final long serialVersionUID = 1L;
  private final static int ASCENDING = 0;
  private final static int DESCENDING = 1;
    /**
   * Main method for testing this class
   *
   * @param argv the command line options
     * @throws Exception 
   */
  public static void main(String [] argv) throws Exception
 {
	   String path,trainingDataFilename;
	   DataSource sourcetrain;
	   Instances data; 
	   Result r = new Result();
	   
       String datasetList[] ={"corel5k","CAL500","bibtex","LLOG","Enron","Birds","Medical","Genebase","SLASHDOT","Birds","Yeast","Flags","Emotions","Scene"};
	   path="/Users/ra12404/Desktop/meka-1.7.5/data/LaCova/Datasets/";
	   String Mydataset;
	   
	   BufferedWriter writer = new BufferedWriter(new FileWriter("/Users/ra12404/Desktop/meka-1.7.5/data/LaCova/Datasets/features.csv"));	
	   	 
	   for (int d=0; d<datasetList.length; d++)
	   {	
		    Mydataset=datasetList[d];
		    writer.write(Mydataset);
	 	    writer.write(',');

		    trainingDataFilename = path+Mydataset+".arff";
			sourcetrain = new DataSource(trainingDataFilename);
			data =  sourcetrain.getDataSet(); 
		    MLUtils.prepareData(data);
		    
		    int L=data.classIndex();
	 	    int f=data.numAttributes();
	 	    int partf=(int) Math.round( (f-L)*0.25);
			writer.write(String.valueOf(partf));
	 	    
		    BinC45ModelSelection model=new BinC45ModelSelection(0,data);
		    //Compute score for each feature
		    double [][] features=model.featuresselection(data);
		    double [][] newfeatures=new double [f-L][2];
		    
		    for(int i=0;i<(f-L);i++)
		    {	    		
		    	newfeatures[i][0]=i+L;
		    	newfeatures[i][1]=features[i+L][1];
		    }		
		    
		    //for(int j=0;j<(f-L);j++)
		    	//System.out.println(newfeatures[j][0]);
		    
		    //Sort the matrix by quality (less is better)
		    sortAttribute(newfeatures,1,ASCENDING);
		    
		    //Select the best 25%
		    int [] featureIndices = featureIndicesbyTPercent(0.25, newfeatures);
		    
		    int [] labels=new int [L];
		    for(int l=0;l<L;l++)
		    	labels[l]=l;		    
		    	
		    int keep[] = A.join(labels,featureIndices);
		    
		    Instances newData= F.remove(data, keep, true);
		    
			/* Writing to file */
		    int index=0;
		    for(int i=0;i<featureIndices.length;i++)
		    {
	    			writer.write("\n");
			 	    writer.write(',');
		    		writer.write(String.valueOf(index++));
		    		writer.write(',');
		    		writer.write(String.valueOf(featureIndices[i]-L));
		    }		    
			writer.write("\n");
	   }
	   writer.flush();
	   writer.close();	  	 
  }
  public static int[] cov_features(Instances data) throws Exception
  {
	    int L=data.classIndex();
	    int f=data.numAttributes();
	    
	    BinC45ModelSelection model=new BinC45ModelSelection(0,data);
	    //Compute score for each feature
	    double [][] features=model.featuresselection(data);
	    double [][] newfeatures=new double [f-L][2];
	    
	    for(int i=0;i<(f-L);i++)
	    {	    		
	    	newfeatures[i][0]=i+L;
	    	newfeatures[i][1]=features[i+L][1];
	    }		
	    
	    //Sort the matrix by quality (less is better)
	    sortAttribute(newfeatures,1,ASCENDING);
	    
	    //Select the best 25%
	    int [] featureIndices = featureIndicesbyTPercent(0.25, newfeatures);	    	
	    
	  return featureIndices;
  }
  
  public static Instances cov_trasform(Instances data) throws Exception
  {
	    //This is for classifier chain assuming only one label
	    int L=data.classIndex();
	    int index=L;
	    
	    if(L==0)
	    	index=L+1;
	    
	    int f=data.numAttributes();
	    System.out.println(f);
	    
	    BinC45ModelSelection model=new BinC45ModelSelection(0,data);
	    
	    //Compute score for each feature
	    double [][] features=model.featuresselectionCC(data, L);
	    double [][] newfeatures=new double [f][2];
	    
	    for(int i=0;i<f;i++)
	    {
	    	if(i!=index)
	    	{
	    	newfeatures[i][0]=i;
	    	newfeatures[i][1]=features[i][1];
	    	}
	    }		
	    
	    //Sort the matrix by quality (less is better)
	    sortAttribute(newfeatures,1,ASCENDING);
	    
		   for(int j=0;j<f;j++)
		    	System.out.println(newfeatures[j][1]);
	    
	    //Select the best 25%
	    int [] featureIndices = featureIndicesbyTPercent(0.25, newfeatures);
	    	    
	    /*int [] labels=new int [L];
	    for(int l=0;l<L;l++)
	    	labels[l]=l;*/	
	    
	    int keepClass[]	=new int[1];
	    keepClass[0]=L;
	    int keep[] = A.join(keepClass, featureIndices);
	    
	    Instances newData= F.remove(data, keep, true);
	    
	  return newData;
  }
  
  public static int[] featureIndicesbyTPercent(double t, double[][] sortedEvaluatedAttributeList) 
  {
      if ((t >= 1) || (t <= 0))
      {
          System.out.println("t should be a value in (0,1)");
          System.exit(1);
      }

      int[] featureIndices = new int[(int) Math.round((double) t * sortedEvaluatedAttributeList.length)];

      for (int i = 0; i < featureIndices.length; i++) 
      {
          featureIndices[i] = ((int) sortedEvaluatedAttributeList[i][0]); //add the index of the ith best feature to the array
      }

      return featureIndices;
  }

  public static void sortAttribute(double[][] ranking, int index, int sortOrder) 
  {

      if (sortOrder == DESCENDING) 
      {
          for (int i = 0; i < ranking.length; i++) 
          {
              int index_max = i;
              for (int j = i + 1; j < ranking.length; j++) 
              {
                  if (ranking[j][index] > ranking[index_max][index]) 
                  {
                      index_max = j;
                  }
              }
              if (index_max != i) 
              {
                  double auxRank = ranking[index_max][1];
                  double auxIndex = ranking[index_max][0];

                  ranking[index_max][0] = ranking[i][0];
                  ranking[index_max][1] = ranking[i][1];
                  ranking[i][1] = auxRank;
                  ranking[i][0] = auxIndex;
              }
          }
      } else if (sortOrder == ASCENDING)
      {
          for (int i = 0; i < ranking.length; i++) 
          {
              int index_max = i;
              for (int j = i + 1; j < ranking.length; j++) 
              {
                  if (ranking[j][index] < ranking[index_max][index]) 
                  {
                      index_max = j;
                  }
              }
              if (index_max != i)
              {
                  double auxRank = ranking[index_max][1];
                  double auxIndex = ranking[index_max][0];

                  ranking[index_max][0] = ranking[i][0];
                  ranking[index_max][1] = ranking[i][1];
                  ranking[i][1] = auxRank;
                  ranking[i][0] = auxIndex;
              }
          }
      }
  }

}

