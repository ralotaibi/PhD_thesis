package weka.ShiftInjection.bias;

import weka.ShiftInjection.basic.Dataset;

public class NewNonLinearShift extends Bias
{	 
    public NewNonLinearShift() 
    { 
    	super(); 
    }   
    /**
     * Minimum allowable value for the number of standard deviations.
     * @return Double.NEGATIVE_INFINITY
     */
    public @Override double getValueMin()
    {
        return Double.NEGATIVE_INFINITY;
    }   
    /**
     * Maximum allowable value for number of standard deviations
     * @return Double.POSITIVE_INFINITY
     */
    public @Override double getValueMax()
    {
        return Double.POSITIVE_INFINITY;
    }   
    public NewNonLinearShift(double value)
    {
        super(value);     
    } 
    
	@Override
    public Dataset injectBias(Dataset dataset, int attribute)
    {
		 if (dataset.getAttribute(attribute).isNumeric())
	        {
	            double attvalue;
	            Dataset copy = dataset.clone();
	            double mean=dataset.getMean(attribute);
	    		double sd=Math.sqrt(dataset.getVariance(attribute));

	            for (int x = 0; x < dataset.numInstances(); x++)
	                {
	                    if (!copy.getInstance(x).isMissing(attribute))
	                    {
	                        attvalue = dataset.getInstance(x).doubleValue(attribute);
	                        double var=attvalue-mean/sd;
	                        var=Math.pow(var,3 );
	                        copy.getInstance(x).setAttributeValue(attribute, (sd*var)+mean);
	                    }
	                }
	            return copy;
	        }
	        else
	            return dataset;
    }
    	   
    /**
     * Units for the parameter of this bias.
     * @return "Standard Deviations"
     */
    public @Override String getUnit(int whichUnit)
    {
    	if (whichUnit==0)
    		return "Standard Deviations";
    	else
    		return "Only one unit in this bias";
    }
    
    public String getLongDescription()
    {
        return "Non Linear Shift";
    }
   
    /**
     * Display-friendly name of this bias.
     * @return "Covariate Shift"
     */
    public String getName() 
    { 
    	return "Non-Linear Shift"; 
    }
    public Dataset injectRandomBias(Dataset original, int attribute) 
    {
		// TODO Auto-generated method stub
		return null;
	}
}
