package weka.ShiftInjection.bias;

import weka.ShiftInjection.basic.*;


public class CubeNonLinearShift extends Bias
{
    public CubeNonLinearShift() 
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
    public CubeNonLinearShift(double value)
    {
        super(value);     
    }    
    /**
     * Introduce a Non-Linear Shift along the given attribute in the given dataset.
     * @param dataset dataset into which to introduce the  shift.
     * @param attribute index of the attribute along which to introduce the shift.
     * @return A clone of the input dataset, with instances altered to reflect the shift.
     */
    public Dataset injectBias(Dataset dataset, int attribute)
    {
    	
        if (dataset.getAttribute(attribute).isNumeric())
        {
            double attvalue;
            Dataset copy = dataset.clone();

            for (int x = 0; x < dataset.numInstances(); x++)
                {
                    if (!copy.getInstance(x).isMissing(attribute))
                    {
                        attvalue = dataset.getInstance(x).doubleValue(attribute);
                        copy.getInstance(x).setAttributeValue(attribute, Math.pow(attvalue,3 ));
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

	@Override
	public Dataset injectRandomBias(Dataset original, int attribute) {
		// TODO Auto-generated method stub
		return null;
	}
}
