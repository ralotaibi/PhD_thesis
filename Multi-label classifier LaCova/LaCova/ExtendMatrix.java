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
 *    SplitCriterion.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *
 */
/*
 *    Extended by Reem Alotaibi to handle multi-label data, 2015
 *	  ra12404@bristol.a.cuk
 */

package meka.classifiers.multilabel.LaCova;
import weka.core.Instances;
import weka.core.matrix.*;
import meka.core.MLUtils;

/**
 * Abstract class for computing splitting criteria
 * with respect to distributions of class values.
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision: 8034 $
 */
public class ExtendMatrix
{

    /** for serialization */
    private static final long serialVersionUID = 5490996638027101259L;
    
    /** Find c_ij */

    protected double[][] find_cij(Instances data, int L)
    {
        double sum=0;
        double p[] = new double[L];
        double pw [][]=new double[L][L];
        double cov [][]=new double[L][L];
        double sum_cov=0;
        int n=data.numInstances();
        
        p=MLUtils.labelCardinalities(data); //return the frequency of each label of dataset data.
        
        for(int j = 0; j < L; j++)
        {
            for(int k=0;k<L;k++)
            {
                for(int i=0;i<n;i++)
                {
                    if((data.instance(i).value(j)==data.instance(i).value(k))&&(data.instance(i).value(k)==1)) //both are 1's
                    {
                        pw[j][k] +=1;
                    }
                }
            }
        }
        
        return pw;
    }

    public Matrix Vector_to_Matrix (double vals[], int m,int n)
    {
        Matrix M1=new Matrix(m,n) ;
        
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                M1.set(i,j,vals[i]);
            }
        }
        return M1;
    }
    
    public double sum_abs (Matrix A)
    {
        double sum = 0;
        int m = A.getRowDimension();
        int n = A.getColumnDimension();
        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < m; i++)
            {
                sum += A.get(i,j);
            }
        }
        return sum;
    }



  }


