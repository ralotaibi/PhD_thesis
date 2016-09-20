package meka.classifiers.multilabel.LaCovaC;

import weka.core.Instances;

 /*
 *    Author Reem Alotaibi to handle multi-label data, 2016
 *	  ra12404@bristol.a.cuk
 */
 
public class FisherExact 
{
	 private static final boolean DEBUG = false;
	 private static double[] f;
	 static int maxSize;
	 
	 public FisherExact(int maxSize) 
	 {
		    this.maxSize=maxSize;
		    double cf=1.0;
		    f=new double[maxSize+1];
		    f[0]=0.0;
		    for(int i=1; i<=this.maxSize; i++)		     
		    	{
		    	f[i]=f[i-1]+Math.log(i);
		    	}
	 }
	 public static int [][]Contingency(Instances data, int [] comb)
	    {

	        int n=data.numInstances();
	        
	        int ConTable[][]=new int[2][2];

	        for(int i = 0; i < n; i++) 
	            if ((int)data.instance(i).value(comb[0])==0)
	                if ((int)data.instance(i).value(comb[1])==0)
	                		ConTable[0][0]=ConTable[0][0]+1;
	                else
	                	ConTable[0][1]=ConTable[0][1]+1;
	            else 
	                if ((int)data.instance(i).value(comb[1])==0)
	                	ConTable[1][0]=ConTable[1][0]+1;
	                else
	                	ConTable[1][1]=ConTable[1][1]+1;
	                
	     return ConTable;
	    }
	 /**
	     * calculates the P-value for this specific state
	     *
	     * @param a     a, b, c, d are the four cells in a 2x2 matrix
	     * @param b
	     * @param c
	     * @param d
	     * @return the P-value
	     */
	    public final static double getP(int a,int b, int c, int d) 
	    {
	        int n=a+b+c+d;	       
	        double p;
	        p = (f[a + b] + f[c + d] + f[a + c] + f[b + d]) - (f[a] + f[b] + f[c] + f[d] + f[n]);
	        return Math.exp(p);
	    }
	    /**
	     * Calculates the one-tail P-value for the Fisher Exact test.  Determines whether to calculate the right- or left-
	     * tail, thereby always returning the smallest p-value.
	     *
	     * @param a     a, b, c, d are the four cells in a 2x2 matrix
	     * @param b
	     * @param c
	     * @param d
	     * @return one-tailed P-value (right or left, whichever is smallest)
	     */
	    public final double getCumlativeP(int a, int b, int c, int d) 
	    {
	        int min, i;
	        int n = a + b + c + d;
	        if (n > maxSize) 
	        {
	            return Double.NaN;
	        }
	        double p = 0;

	        p += getP(a, b, c, d);
	        if (DEBUG) 
	        {
	        	System.out.println("p = " + p);
	        }
	        if ((a * d) >= (b * c)) 
	        {
	            if (DEBUG)
	            {
	            	System.out.println("doing R-tail: a=" + a + " b=" + b + " c=" + c + " d=" + d);}
	            min = (c < b) ? c : b;
	            for (i = 0; i < min; i++) 
	            {
	                if (DEBUG)
	                {
	                	System.out.print("doing round " + i);
	                	}
	                p += getP(++a, --b, --c, ++d);
	                if (DEBUG) 
	                {
	                	System.out.println("\ta=" + a + " b=" + b + " c=" + c + " d=" + d);
	                }
	            }
	            System.out.println("");
	        }
	        if ((a * d) < (b * c)) 
	        {
	            if (DEBUG) 
	            {
	            	System.out.println("doing L-tail: a=" + a + " b=" + b + " c=" + c + " d=" + d);
	            }
	            min = (a < d) ? a : d;
	            for (i = 0; i < min; i++) 
	            {
	                if (DEBUG) {System.out.print("doing round " + i);
	                }
	                double pTemp = getP(--a, ++b, ++c, --d);
	                if (DEBUG) 
	                {
	                	System.out.print("\tpTemp = " + pTemp);
	                }
	                p += pTemp;
	                if (DEBUG) {System.out.println("\ta=" + a + " b=" + b + " c=" + c + " d=" + d);}
	            }
	        }
	        return p;
	    }

	    /**
	     * Calculates the right-tail P-value for the Fisher Exact test.
	     *
	     * @param a     a, b, c, d are the four cells in a 2x2 matrix
	     * @param b
	     * @param c
	     * @param d
	     * @return one-tailed P-value (right-tail)
	     */
	    public final static double getRightTailedP(int a, int b, int c, int d)
	    {
	        int min, i;
	        int n = a + b + c + d;
	        if (n > maxSize) 
	        {
	            return Double.NaN;
	        }
	        double p = 0;

	        p += getP(a, b, c, d);
	        if (DEBUG) {System.out.println("p = " + p);}
	        if (DEBUG) {System.out.println("doing R-tail: a=" + a + " b=" + b + " c=" + c + " d=" + d);}
	        min = (c < b) ? c : b;
	        for (i = 0; i < min; i++) 
	        {
	            p += getP(++a, --b, --c, ++d);

	        }
	        return p;
	    }

	    /**
	     * Calculates the left-tail P-value for the Fisher Exact test.
	     *
	     * @param a     a, b, c, d are the four cells in a 2x2 matrix
	     * @param b
	     * @param c
	     * @param d
	     * @return one-tailed P-value (left-tail)
	     */
	    public final static double getLeftTailedP(int a, int b , int c, int d) 
	    {
	        int min, i;
	        int n = a + b + c + d;
	        if (n > maxSize) 
	        {
	            return Double.NaN;
	        }
	        double p = 0;

	        p += getP(a, b, c, d);
	        if (DEBUG) {System.out.println("p = " + p);}
	        if (DEBUG) {System.out.println("doing L-tail: a=" + a + " b=" + b + " c=" + c + " d=" + d);}
	        min = (a < d) ? a : d;
	        for (i = 0; i < min; i++) {
	            if (DEBUG) {System.out.print("doing round " + i);}
	            double pTemp = getP(--a, ++b, ++c, --d);
	            if (DEBUG) {System.out.print("\tpTemp = " + pTemp);}
	            p += pTemp;
	            if (DEBUG) {System.out.println("\ta=" + a + " b=" + b + " c=" + c + " d=" + d);}
	        }
	        return p;
	    }
	    /**
	     *   Calculates the two-tailed P-value for the Fisher Exact test.
	     *
	     *   In order for a table under consideration to have its p-value included
	     *   in the final result, it must have a p-value less than the original table's P-value, i.e.
	     *   Fisher's exact test computes the probability, given the observed marginal
	     *   frequencies, of obtaining exactly the frequencies observed and any configuration more extreme.
	     *   By "more extreme," we mean any configuration (given observed marginals) with a smaller probability of
	     *   occurrence in the same direction (one-tailed) or in both directions (two-tailed).
	     *
	     * @param a     a, b, c, d are the four cells in a 2x2 matrix
	     * @param b
	     * @param c
	     * @param d
	     * @return two-tailed P-value
	     */
	    public final static double getTwoTailedP(int a, int b, int c, int d) 
	    {
	        int min, i;
	        int n = a + b + c + d;
	        if (n > maxSize) 
	        {
	            return Double.NaN;
	        }
	        double p = 0;

	        double baseP = getP(a, b, c, d);

	        if (DEBUG) {System.out.println("baseP = " + baseP);}
	        int initialA = a, initialB = b, initialC = c, initialD = d;
	        p += baseP;
	        if (DEBUG) {System.out.println("p = " + p);}
	        if (DEBUG) {System.out.println("Starting with R-tail: a=" + a + " b=" + b + " c=" + c + " d=" + d);}
	        min = (c < b) ? c : b;
	        for (i = 0; i < min; i++) {
	            if (DEBUG) {System.out.print("doing round " + i);}
	            double tempP = getP(++a, --b, --c, ++d);
	            if (tempP <= baseP) {
	                if (DEBUG) {System.out.print("\ttempP (" + tempP + ") is less than baseP (" + baseP + ")");}
	                p += tempP;
	            }
	            if (DEBUG) {System.out.println(" a=" + a + " b=" + b + " c=" + c + " d=" + d);}
	        }

	        // reset the values to their original so we can repeat this process for the other side
	        a = initialA;
	        b = initialB;
	        c = initialC;
	        d = initialD;

	        if (DEBUG) {System.out.println("Now doing L-tail: a=" + a + " b=" + b + " c=" + c + " d=" + d);}
	        min = (a < d) ? a : d;
	        if (DEBUG) {System.out.println("min = " + min);}
	        for (i = 0; i < min; i++) {
	            if (DEBUG) {System.out.print("doing round " + i);}
	            double pTemp = getP(--a, ++b, ++c, --d);
	            if (DEBUG) {System.out.println("  pTemp = " + pTemp);}
	            if (pTemp <= baseP) {
	                if (DEBUG) {System.out.print("\ttempP (" + pTemp + ") is less than baseP (" + baseP + ")");}
	                p += pTemp;
	            }
	            if (DEBUG) {System.out.println(" a=" + a + " b=" + b + " c=" + c + " d=" + d);}
	        }
	        return p;
	    }
	    
	    public final static double Fisher_pvalue(Instances D, int[] comb)
	    {
	    	int [][] Cont=Contingency(D, comb);
	    	int a=Cont[0][0];
	    	int b=Cont[0][1];

	    	int c=Cont[1][0];

	    	int d=Cont[1][1];

	    	
	    	double twoTailedP =getTwoTailedP(a,b,c,d);
	    	//double leftP =getLeftTailedP(a,b,c,d);
	    	//double rightP =getRightTailedP(a,b,c,d);

			return twoTailedP;
	    	
	    }
	    
	 
}
