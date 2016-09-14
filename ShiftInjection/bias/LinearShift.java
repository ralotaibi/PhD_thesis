package weka.ShiftInjection.bias;

import weka.ShiftInjection.basic.Dataset;
import weka.ShiftInjection.basic.Instance;
import weka.ShiftInjection.basic.myRandomizer;

public class LinearShift extends Bias 
{
    private static final String MULT = "mult";
    private static final String ADD = "add";
	
    public LinearShift() 
    {
    	super(); numUnits=2;
    }
    
    public LinearShift(double degree) 
    {
    	super(degree); numUnits=2;
    }    
    /*
     * @param mean is used as the mean for the Gaussian distribution
     * @param std is the standard deviation
     */
    public LinearShift(double mult, double add)
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
		Dataset ret = dataset.emptyClone();
		for (int i=0; i<dataset.numInstances(); i++)
		{
			Instance instance = dataset.getInstance(i);
			instance.setAttributeValue(attribute, instance.doubleValue(attribute)*getValue(MULT)+getValue(ADD));
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
		

		double mult = 1/(myRandomizer.generator.nextDouble()*getValue()+getValue());

		//if (myRandomizer.generator.nextBoolean())
			//mult = -1*mult;

		double range = mult*original.getMax(attribute)-original.getMin(attribute);

		double add = (1-getValue())*myRandomizer.generator.nextDouble()*range;//-range/2;
		System.out.println("a="+mult+"b= "+add);
		setProperty(MULT,mult);
		setProperty(ADD,add);
		return injectBias(original,attribute);
	}
}
