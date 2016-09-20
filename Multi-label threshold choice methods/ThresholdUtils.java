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

package meka.core;

import weka.core.*;
import java.util.*;

/**
 * ThresholdUtils - Helpful functions for calibrating thresholds.
 * @author Jesse Read (jesse@tsc.uc3m.es)
 * @version	March 2013
 */
public abstract class ThresholdUtils {

	/**
	 * ThresholdStringToArray - parse a threshold option string to an array of L thresholds (one for each label variable).
	 */
	public static double[] thresholdStringToArray(String top, int L) {
		if (top.startsWith("[")) {
			//if (L != 
			return MLUtils.toDoubleArray(top);							// threshold vector       [t1 t2 ... tL]]
		}
		else {
			double t[] = new double[L];
			Arrays.fill(t,Double.parseDouble(top));					// make a threshold vector [t t t ... t]
			return t;
		}
	}

	/**
	 * CalibrateThreshold - Calibrate a threshold using PCut: the threshold which results in the best approximation of the label cardinality of the training set.
	 * @param	Y			labels
	 * @param	LC_train	label cardinality of the training set
	 */
	public static double calibrateThreshold(ArrayList<double[]> Y, double LC_train) { 

		if (Y.size() <= 0) 
			return 0.5;

		int N = Y.size();
		ArrayList<Double> big = new ArrayList<Double>();
		for(double y[] : Y) {
			for (double y_ : y) {
				big.add(y_);
			}
		}
		Collections.sort(big);

		int i = big.size() - (int)Math.round(LC_train * (double)N);

		if (N == big.size()) { // special cases
			if (i+1 == N) // only one!
				return (big.get(N-2)+big.get(N-1)/2.0);
			if (i+1 >= N) // zero!
				return 1.0;
			else
				return Math.max(((double)(big.get(i)+big.get(i+1))/2.0), 0.00001);
		}

		return Math.max(((double)(big.get(i)+big.get(Math.max(i+1,N-1))))/2.0 , 0.00001);
	}

	/**
	 * CalibrateThreshold - Calibrate a vector of thresholds (one for each label) using PCut: the threshold t[j] which results in the best approximation of the frequency of the j-th label in the training data.
	 * @param	Y			labels
	 * @param	LC_train[]	average frequency of each label
	 */
	public static double[] calibrateThresholds(ArrayList<double[]> Y, double LC_train[]) { 

		int L = LC_train.length;
		double t[] = new double[L];

		ArrayList<double[]> Y_[] = new ArrayList[L];
		for(int j = 0; j < L; j++) {
			Y_[j] = new ArrayList<double[]>();
		}

		for(double y[] : Y) {
			for(int j = 0; j < L; j++) {
				Y_[j].add(new double[]{y[j]});
			}
		}

		for(int j = 0; j < L; j++) {
			t[j] = calibrateThreshold(Y_[j],LC_train[j]);
		}

		return t;
	}

	/**
	 * Threshold - returns the labels after the prediction-confidence vector is passed through a vector of thresholds.
	 * @param	Rpred[][]	label confidence predictions in [0,1]
	 * @param	t[]			threshold for each label
	 */
	public static final int[][] threshold(double Rpred[][], double t[]) {
		int Ypred[][] = new int[Rpred.length][Rpred[0].length];
		for(int i = 0; i < Rpred.length; i++) {
			for(int j = 0; j < Rpred[i].length; j++) {
				Ypred[i][j] = (Rpred[i][j] >= t[j]) ? 1 : 0;
			}
		}
		return Ypred;
	}

	/**
	 * Threshold - returns the labels after the prediction-confidence vector is passed through threshold.
	 * @param	Rpred[][]	label confidence predictions in [0,1]
	 * @param	t			threshold
	 */
	public static final int[][] threshold(double Rpred[][], double t) {
		int Ypred[][] = new int[Rpred.length][Rpred[0].length];
		for(int i = 0; i < Rpred.length; i++) {
			for(int j = 0; j < Rpred[i].length; j++) {
				Ypred[i][j] = (Rpred[i][j] >= t) ? 1 : 0;
			}
		}
		return Ypred;
	}

	/**
	 * Threshold - returns the labels after the prediction-confidence vector is passed through threshold(s).
	 * @param	rpred[]	label confidence predictions in [0,1]
	 * @param	ts		threshold String
	 */
	public static final int[] threshold(double rpred[], String ts) {
		int L = rpred.length;
		double t[] = thresholdStringToArray(ts,L);
		int ypred[] = new int[L];
		for(int j = 0; j < L; j++) {
			ypred[j] = (rpred[j] >= t[j]) ? 1 : 0;
		}
		return ypred;
	}
	/**
	 * Round - simply round numbers (e.g., 2.0 to 2) -- for multi-target data (where we don't *yet* use a threshold).
	 * @param	Rpred[][]	class predictions in [0,1,...,K]
	 * @return  integer representation of the predictions
	 */
	public static final int[][] round(double Rpred[][]) {
		int Ypred[][] = new int[Rpred.length][Rpred[0].length];
		for(int i = 0; i < Rpred.length; i++) {
			for(int j = 0; j < Rpred[i].length; j++) {
				Ypred[i][j] = (int)Math.round(Rpred[i][j]);
			}
		}
		return Ypred;
	}
	 //Reeeem
    public static final int[][] thresholdPCut(double Rpred[][], double t[])
    {
		int Ypred[][] = new int[Rpred.length][Rpred[0].length];
		for(int i = 0; i < Rpred.length; i++)
        {
			for(int j = 0; j < Rpred[i].length; j++)
            {
				Ypred[i][j] = (Rpred[i][j] > t[j]) ? 1 : 0;
			}
		}
		return Ypred;
	}
    //Reeeem
    public static final int[][] thresholdRCut(double Rpred[][], double t[])
    {
		int Ypred[][] = new int[Rpred.length][Rpred[0].length];
		for(int i = 0; i < Rpred.length; i++)
        {
			for(int j = 0; j < Rpred[i].length; j++)
            {
				Ypred[i][j] = (Rpred[i][j] >= t[i]) ? 1 : 0;
			}
		}
		return Ypred;
	}
	 /**
	 * CalibrateThreshold - Calibrate a vector of thresholds (one for each label) using RCut: the threshold t[j] which results in the best approximation of the frequency of the j-th label in the training data.
	 * @param	Y			labels
	 * @param	k	average frequency of D
	 */
	public static double[] calibrateThresholdsRCut(double  scores [][], int k)
    {
        int N = scores.length;
        int L = scores[0].length;
		double t[] = new double[N];//instance-wise
        double [] instscores;
        double temp;
        
        for(int i=0;i<N;i++)
        {
            instscores=new double[L];
            for(int j=0;j<L;j++)
            {
                instscores[j]=scores[i][j];
            }
            
            Arrays.sort(instscores);
            
            for( int m = 0; m < instscores.length/2; ++m )
            {
                temp = instscores[m];
                instscores[m] = instscores[instscores.length - m - 1];
                instscores[instscores.length - m - 1] = temp;
            }
            if (k==0)
                t[i]=1.5;
            else
                t[i]=instscores[k-1];
        }
        
		return t;
	}
    
    //Reeeem
    public static final int[][] thresholdMCut(double Rpred[][], double t[])
    {
		int Ypred[][] = new int[Rpred.length][Rpred[0].length];
		for(int i = 0; i < Rpred.length; i++)
        {
			for(int j = 0; j < Rpred[i].length; j++)
            {
				Ypred[i][j] = (Rpred[i][j] >= t[i]) ? 1 : 0;
			}
		}
		return Ypred;
	}
	 /**
	 * CalibrateThreshold - Calibrate a vector of thresholds (one for each label) using RCut: the threshold t[j] which results in the best approximation of the frequency of the j-th label in the training data.
	 * @param	Y			labels scores
	 */
	public static double[] calibrateThresholdsMCut(double  scores [][])
    {
        int N = scores.length;
        int L = scores[0].length;
		double t[] = new double[N]; //instance-wise
        double [] instscores;
        double temp, diff, Maxdiff=-1;
        int a=0,b=1;
        
        for(int i=0;i<N;i++)
        {
            instscores=new double[L];
            for(int j=0;j<L;j++)
            {
                instscores[j]=scores[i][j];
            }
            
            Arrays.sort(instscores);
            
            for( int m = 0; m < instscores.length/2; ++m )
            {
                temp = instscores[m];
                instscores[m] = instscores[instscores.length - m - 1];
                instscores[instscores.length - m - 1] = temp;
            }
            
            for( int m = 0; m < instscores.length-1; ++m )
            {
                diff=instscores[m]-instscores[m+1];
                if (diff>=Maxdiff)
                {
                    a=m;
                    b=m+1;
                    Maxdiff=diff;
                }

            }

                t[i]=(instscores[a]+instscores[b])/2;
        }
        
		return t;
	}
	 //Reeeem
	  public static double calibrateThresholdSCut(double  scores [][], int actual [][], double cost) throws Exception //each time threshold per label
    {
        int N = scores.length;
        int L = scores[0].length;
        int total=N*L;
        double FN,FP;
        
        // sorting the confidences and set initial threshohlds for all labels
        //Arrays.sort(scores);
        double t = 0.5;
        
        double[] measureTable = new double[3];
        double counter = 0;
        double tempThreshold = 0;
        int conv = 0;
        double[] performance = new double[total+1];
        double []Loss=new double[total+1];
        double []Allscores=new double[total];
        int []Allactual=new int[total];
        
        //put all scores in one bin and find one global threshold
        for (int i=0;i<N;i++)
            for(int j=0;j<L;j++)
            {
                Allscores[i*j]=scores[i][j];
                Allactual[i*j]=actual[i][j];
            }
        
        double tm = 0; int ty=0;
        for(int i = 0;i<total;i++)
        {
            for(int j = (total-1);j>=(i+1);j--)
            {
                if(Allscores[j]<Allscores[j-1])
                {
                    tm = Allscores[j];
                    Allscores[j]=Allscores[j-1];
                    Allscores[j-1]=tm;
                    ty = Allactual[j];
                    Allactual[j]=Allactual[j-1];
                    Allactual[j-1]=ty;
                }
            }
        }
        double score = 0;         
        //get a measure for all Thresholds
        for (int l = total ; l >= 0; l--) //posa instances diladi tosa thresshold
        {
            if (l == 0)
            {
                t = Allscores[l];
            }
            else if (l==total)
            {
                t = Allscores[l-1]+1.5;
            }
            else
            {
                t = (Allscores[l] + Allscores[l - 1]) / 2;
            }
            //get the predicted labels for all instances according to Thresholds
            int[] predictedLabels = new int[total];
            for (int k = 0; k < total; k++)
            {
                predictedLabels[k] = (Allscores[k]>= t) ? 1 : 0;
            }
            FN=Metrics.P_FalseNegatives(Allactual, predictedLabels);
            FP=Metrics.P_FalsePositives(Allactual, predictedLabels);
            Loss[l]=2*(cost*(FN/total)+(1-cost)*(FP/total));
            score+=Loss[l];
        }
        for (int i = 0; i <= total; i++)
        {
            performance[i] = Math.abs(0 - Loss[i]);
        }
        int c = Utils.minIndex(performance);
        if (c == 0)
        {
            t = Allscores[c];
        }
        else if(c==total)
        {
            t= Allscores[c-1]+1.5;
            
        }
        else
        {
            t = (Allscores[c] + Allscores[c - 1]) / 2;
        }
        
        
        return t;
        
    }
    //Reem
    /**
	 * CalibrateThreshold - Calibrate a vector of thresholds (one for each label) using RCut: the threshold t[j] which results in the best approximation of the frequency of the j-th label in the training data.
	 * @param	Y			labels scores
     * @param	actual		actual labels
	 */
	public static double calibrateThresholdsSCut(double  scores [], int actual [], double cost) throws Exception //each time threshold per label
    {
        int N = scores.length;
        double FN,FP;
        double []Loss=new double[N+1];
        
        // sorting the confidences and set initial threshohlds for all labels
            //Arrays.sort(scores);
            double t = 0.5;
        
        double[] measureTable = new double[3];
        double counter = 0;
        double tempThreshold = 0;
        int conv = 0;
        double[] performance = new double[N+1];
        
        double tm = 0; int ty=0;
        for(int i = 0;i<N;i++)
        {
            for(int j = (N-1);j>=(i+1);j--)
            {
                if(scores[j]<scores[j-1])
                {
                    tm = scores[j];
                    scores[j]=scores[j-1];
                    scores[j-1]=tm;
                    ty = actual[j];
                    actual[j]=actual[j-1];
                    actual[j-1]=ty;

                }
            }
        }
                double score = 0;
                //get a measure for all Thresholds
                for (int l = N ; l >= 0; l--) //posa instances diladi tosa thresshold
                {
                    if (l == 0)
                    {
                        t = scores[l];
                    }
                    else if (l==N)
                    {
                        t = scores[l-1]+1.5;
                    }
                    else
                    {
                        t = (scores[l] + scores[l - 1]) / 2;
                    }
                    //get the predicted labels for all instances according to Thresholds
                    int[] predictedLabels = new int[N];
                    for (int k = 0; k < N; k++)
                    {
                            predictedLabels[k] = (scores[k]>= t) ? 1 : 0;
                    }
                    FN=Metrics.P_FalseNegatives(actual, predictedLabels);
                    FP=Metrics.P_FalsePositives(actual, predictedLabels);
                    Loss[l]=2*(cost*(FN/N)+(1-cost)*(FP/N));
                    score+=Loss[l];
                }
                for (int i = 0; i <= N; i++)
                {
                    performance[i] = Math.abs(0 - Loss[i]);
                }
                int c = Utils.minIndex(performance);
                if (c == 0)
                {
                    t = scores[c];
                }
                else if(c==N)
                {
                    t= scores[c-1]+1.5;

                }
                else
                {
                    t = (scores[c] + scores[c - 1]) / 2;
                }
                
        
        return t;
        
    }
    
}
