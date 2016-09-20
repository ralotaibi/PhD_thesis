package meka.classifiers.multilabel.LaCova;

 /*
 *    Extended by Reem Alotaibi to handle multi-label data, 2015
 *	  ra12404@bristol.a.cuk
 */
public class pairCLass implements Comparable
{
	
		  int[] pair;
		  double cov;
		  
		  public pairCLass(int[] A , double c)
		  {
			  pair=A;
			  cov=c;
		  }
		  public double getCov()
		  {
			  return cov;
		  }
		  public int[] getPair()
		  {
			  return pair;
		  }
		  public int compareTo(Object otherPair) 
		  {
		        if( otherPair == null ) 
		        {
		            throw new NullPointerException();
		        }
		        if( !( otherPair instanceof pairCLass)) 
		        {
		            throw new ClassCastException("Invalid object");
		        }
		        Double value = ( (pairCLass) otherPair ).getCov();
		        if(  this.getCov() > value )
		            return 1;
		        else if ( this.getCov() < value )
		            return -1;
		        else
		            return 0;
		 }	   
}
