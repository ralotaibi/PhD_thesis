package weka.ShiftInjection.bias;

import weka.ShiftInjection.basic.Dataset;
import weka.ShiftInjection.basic.Instance;

public class NewLinearShift extends Bias 
{
    private static final String MULT = "mult";
    private static final String ADD = "add";
	
    public NewLinearShift() 
    {
    	super(); numUnits=2;
    }
    
    public NewLinearShift(double degree) 
    {
    	super(degree); numUnits=2;
    }    
    /*
     * @param mean is used as the mean for the Gaussian distribution
     * @param std is the standard deviation
     */
    public NewLinearShift(double mult, double add)
    {
        super();
        setValue(MULT,mult);
        setValue(ADD,add);
        numUnits = 2;
    }
	@Override
	public String getLongDescription() 
	{
		return "I do not feel like writing this right now so there it is.";
	}

	@Override
	public String getName()
	{
		return "Domain Shift";
	}

	@Override
	public String getUnit(int whichUnit) 
	{
		if (whichUnit==0)
			return "Multiplication factor";
		else if (whichUnit==1)
			return "Addition factor";
		else
    		return "Only two units in this bias";
	}

	@Override
	public Dataset injectBias(Dataset dataset, int attribute) 
	{
		double mean=dataset.getMean(attribute);
		double sd=Math.sqrt(dataset.getVariance(attribute));
		
		double mult=Math.pow(2, getPHI());
		double add=(1-Math.pow(2, getPHI()))*mean+(getGAMA()*sd);
		
		Dataset ret = dataset.emptyClone();
		for (int i=0; i<dataset.numInstances(); i++)
		{
			Instance instance = dataset.getInstance(i);
			instance.setAttributeValue(attribute, instance.doubleValue(attribute)*mult+add);
			ret.addInstance(instance);
		}		
		return ret;
	}

    public @Override String toString()
    {
    	return getName()+", f(x)=x*("+getProperty(MULT)+")+("+getProperty(ADD)+")";
    }

	@Override
	public Dataset injectRandomBias(Dataset original, int attribute) 
	{
		double mean=original.getMean(attribute);
		double sd=Math.sqrt(original.getVariance(attribute));
		//System.out.println("mean="+mean+" and sd="+sd);
		double mult=Math.pow(2, getPHI());
		double add=(1-Math.pow(2, getPHI()))*mean+(getGAMA()*sd);
		//System.out.println("a="+mult+" and b= "+add);
		setProperty(MULT,mult);
		setProperty(ADD,add);
		return injectBias(original,attribute);
	}
}
