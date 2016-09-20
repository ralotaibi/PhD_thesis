package meka.classifiers.multilabel.LaCovaC;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.Set;

import meka.classifiers.multilabel.LaCovaC.CorrelationMatrix;
import meka.classifiers.multilabel.LaCovaC.pairCLass;
import meka.core.A;
import meka.core.F;
import meka.core.MLUtils;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

 /*
 *    Author Reem Alotaibi to handle multi-label data, 2016
 *	  ra12404@bristol.a.cuk
 */
 
public class ClusteringCorr 
{		
	    protected  LabelList clus = new LabelList();

		public  LabelList generate_clusters(Instances data,int L, String linkage,int []inx) throws IOException, ClassNotFoundException
		{
			clus=new LabelList();
			
			CorrelationMatrix CM=new CorrelationMatrix();
	        
		    //find_CorrMatrix is the upper triangular correlation matrix
	        double [][]COR_Mat=CM.find_CorrMatrix(data,L);	           		
	        
		    //find_CM is the upper triangular
	         double [][]Dist=CM.find_distance(COR_Mat,L);
	        
			pairCLass[] labelPairs;//Cov_Mat

			int[][] currClusters;
			int[][] newClusters = null;
			    			
			// build initial combination set (each label in a separate group)
			currClusters = new int[L][1];
			for (int i = 0; i < L; i++) 
			{
				currClusters[i][0] =i;
			}
			
			//String representation
			int[][] strClusters = new int[L][1];
			for (int i = 0; i < L; i++) 
			{
				strClusters[i][0] = inx[i];
			}
			
			//take next labels pair, create new combination and build a model
			labelPairs=calculateDependence(Dist);

			double best;			
			
			for (pairCLass pair : labelPairs) 
			{
				double score = pair.getCov();
				int[] comb = pair.getPair();
				//System.out.println("labels:"+comb[0]+" "+comb[1]);
				int length = currClusters.length;
					if (length == 1) 
					{
						//System.out.println("All labels are in the same group. Stop the clustering process!");
						break; // no more combinations possible - stop the process
					}
				
				double criticalValue=CM.find_thr_pair_correlation(data,comb) ;								    
				
				//System.out.println("score=" + score+ " criticalValue=" + criticalValue);

				if (score> 1-criticalValue )
				{
					//System.out.println("Labels are independent!!");

					//System.out.println("Pairs dependence score: " + score+ " is below the criticalValue: " + criticalValue);
					break; 
				}
				else
				{				
					// construct new label set partition
					
					newClusters = buildCombinationSet(currClusters, comb);
					for (int[] newCluster : newClusters) 
					{ // sort the labels within each group
						Arrays.sort(newCluster);
					}
					currClusters = newClusters;
					//System.out.println("Returning  the final labels partition: "+ Clustering.partitionToString(currClusters) + '\n');
					
					}					
			}				
			
			clus.indices=currClusters.clone();
			clus.names=currClusters.clone();
	        //System.out.println("Returning  the final labels partition (indices): "+ ClusteringCorr.partitionToString(clus.indices) + '\n');

			int [][]cls=new int [clus.names.length][1];

			for(int j=0;j<clus.names.length;j++)
			{
				int[] aGroup=clus.names[j].clone();

					for(int i=0;i<aGroup.length;i++)
					{
						for(int k=0;k<strClusters.length;k++)
						{
							if(aGroup[i]==k)
							{
								aGroup[i]=strClusters[k][0];break;
							}
						}
					}				
				cls[j]=aGroup;				
			}
								
			//System.out.println("Returning  the final labels partition (indices): "+ ClusteringCorr.partitionToString(clus.indices) + '\n');
			clus.names=cls;		
			//System.out.println("Returning  the final labels partition (original indices): "+ ClusteringCorr.partitionToString(clus.names) + '\n');
			
			return clus;
		}
		
		
	/**
	* Return pairs of the correlation matrix.
	*
	* @param CM correlation matrix 
	* @return an array of label pairs sorted in descending order of the correlation value
	* @throws Exception if something goes wrong
	*/
	public  pairCLass[] calculateDependence(double [][] CM)
	{
	pairCLass pairObj = null ;
	pairCLass []pairList;
	List<pairCLass> CMList = new ArrayList<pairCLass>();
	  for(int i=0; i<CM.length-1; i++)
	  {
	      for(int j=i+1; j<CM[i].length; j++)
	      {
	          int[] pairs = new int[2];
	          pairs[0] = i;
	          pairs[1] = j;
	          double val = CM[i][j];
	          pairObj = new pairCLass(pairs , val) ;   
	          //System.out.println("pair0="+(pairs[0]+1)+" pair1="+(pairs[1]+1)+" cov="+val);
	          CMList.add(pairObj);
	      }
	  }

	  pairList = new pairCLass[CMList.size()];
	  CMList.toArray(pairList);
	  Arrays.sort(pairList);//, Collections.reverseOrder());
	  for(int i=0;i<pairList.length;i++)
		  {
		  pairCLass x=pairList[i];
		  int []c=x.getPair();
          //System.out.println("pair0="+(c[0])+" pair1="+(c[1])+" cov="+x.getCov());
		  }

	return pairList;
	}
	/**
	* Clusters a new pair of labels and integrates the new group into the given labels partition.
	* 
	* @param partition - label set partition
	* @param pair - labels pair
	* @return a new partition with clustered labels of the pair
	*/
	private static int[][] buildCombinationSet(int[][] partition, int[] pair) 
	{
		
	int[][] newClusters = new int[partition.length - 1][];
	int[][] tmpClusters = new int[partition.length][];
	int i1 = -1;
	int i2 = -1;
	for (int i = 0; i < partition.length; i++) 
	{ // identify indexes of pair values in the
		// partition
		for (int j = 0; j < partition[i].length; j++) 
		{
			if (partition[i][j] == pair[0]) 
			{
				i1 = i;
			}
			if (partition[i][j] == pair[1]) 
			{
				i2 = i;
			}
		}
	}
	if (i1 == i2) // if both labels already in the same set -> there is no change
		return partition;
	for (int k = 0; k < partition.length; k++) 
	{ // copy unchanged sets and unify sets with
		// values from pair
		if (i1 > i2) 
		{ // ensure that i1 is index of first occurrence of one of the values from
			// pair
			int temp = i1;
			i1 = i2;
			i2 = temp;
		}
		if (k != i1) 
		{ // if set's values not in pair -> copy as is
			tmpClusters[k] = partition[k];
		} 
		else 
		{ // set new set to be a union of two previous sets
			tmpClusters[k] = new int[partition[i1].length + partition[i2].length];
			int m;
			for (m = 0; m < partition[i1].length; m++) 
			{
				tmpClusters[k][m] = partition[i1][m];
			}
			int n;
			for (n = 0; n < partition[i2].length; n++) 
			{
				tmpClusters[k][m + n] = partition[i2][n];
			}
		}
	}
	// delete the set which labels were added to another set:
	System.arraycopy(tmpClusters, 0, newClusters, 0, i2);
	// move all sets appearing after eliminated set into one index smaller
	System.arraycopy(tmpClusters, i2 + 1, newClusters, i2, newClusters.length - i2);
	return newClusters;
	}
	/**
	* Returns a string representation of the labels partition.
	* 
	* @param partition - a label set partition
	* @return a string representation of the labels partition
	*/
	public static String partitionToString(int[][] partition) 
	{
	StringBuilder result = new StringBuilder();
	for (int[] aGroup : partition) 
	{
		result.append(Arrays.toString(aGroup));
		result.append(", ");
	}
	return result.toString();
	}
	/**
	* Returns LabelPowerset LP representation of the labels partition.
	* 
	* @param D - Instances
	* @param clusters - a label set partition
	* @return An array of instances (LP) partitioned according to the clusters
	*/
	public Instances[] convertLP(Instances D, int [][] clusters) throws Exception
	{
		Instances [] insClus=new Instances[clusters.length];
		int inx=0;
		int L=D.classIndex();
		for(int [] aGroup:clusters)
		{
			insClus[inx]=new Instances(D);
			insClus[inx]=F.keepLabels(insClus[inx],L,aGroup);
			inx=inx+1;
		}
		
		inx=0;
		
		for (int[] aGroup : clusters) 
		{
			String result = "";
			L=insClus[inx].classIndex();
			// Gather combinations
			HashSet<String> UniqueValues =  new HashSet<String>();
			FastVector ClassValues = new FastVector(L);
			for (int i = 0; i < insClus[inx].numInstances(); i++)
			{
				UniqueValues.add(MLUtils.toBitString(insClus[inx].instance(i),L));
			}
			Iterator<String> it = UniqueValues.iterator();
			while (it.hasNext()) 
			{
				ClassValues.addElement(it.next());
			}
			Attribute classAttribute = new Attribute("Class", ClassValues);
			
			insClus[inx].insertAttributeAt(classAttribute, 0);
			insClus[inx].setClassIndex(0);

			for(int i = 0; i < insClus[inx].numInstances(); i++)
			{
				result="";
				//System.out.println("in="+insClus[inx].instance(i).toString());
				
				for(int x:aGroup)
				{					
					result=result+(String)((int)Math.round(D.instance(i).value(x))+"");
				}	
				//System.out.println("result="+result);
			insClus[inx].instance(i).setClassValue(result);
			}
		
		for (int i = 0; i < L; i++)
			insClus[inx].deleteAttributeAt(1);

		inx+=1;
	}
	return insClus;
}
	
	/**
	* Returns Instances splitted as the labels partition.
	* 
	* @param D - Instances
	* @param clusters - a label set partition
	* @return An array of instances partitioned according to the clusters
	*/
	public Instances[] convert(Instances D, int [][] clusters) throws Exception
	{
		Instances [] insClus=new Instances[clusters.length];
		int inx=0;
		int L=D.classIndex();
		for(int [] aGroup:clusters)
		{
			insClus[inx]=new Instances(D);
			insClus[inx]=F.keepLabels(insClus[inx],L,aGroup);
			insClus[inx].setClassIndex(aGroup.length);
			inx=inx+1;
		}
		/*for(int i=0;i<insClus.length;i++)	
		{
			System.out.println(insClus[i].classIndex());
		}*/
	return insClus;
    }	
	/**
	* Returns Instance splitted as the labels partition.
	* 
	* @param x - Instance
	* @param clusters - a label set partition
	* @return An instance partitioned according to the clusters
	*/
	public Instance convertInstance(Instance x, Instances template) throws Exception
	{
		Instance ins=(Instance) x.copy();
		ins.setDataset(null);
		int inx=0;
		int L=x.classIndex();
		
		for (int i = 0; i < L; i++)
			ins.deleteAttributeAt(0);
		
		ins.insertAttributeAt(0);
		ins.setDataset(template);
		
	return ins;
    }
	
	// join 'a' and 'b' together into 'c' [1,2],[3] -> [1,2,3]
	public double[] join(double a[], double b[]) 
	{
		double c[] = new double[a.length+b.length];
		int i = 0;
		for(int j = 0; j < a.length; j++, i++) 
		{
			c[i] = a[j];
		}
		for(int j = 0; j < b.length; j++, i++) 
		{
			c[i] = b[j];
		}
		return c;
	}

}