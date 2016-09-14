package weka.ShiftInjection.bias;

import java.util.TreeMap;
import weka.ShiftInjection.basic.Dataset;
import weka.ShiftInjection.basic.Utils;


public abstract class Bias
{
	protected static final String DEGREE = "DEGREE";
	protected static final String PHI = "PHI";
	protected static final String GAMA = "GAMA";

    protected TreeMap<String, Object> properties = new TreeMap<String, Object>();
    protected int numUnits = 1;
    
    protected static final double degreeRange=1.5;
    
    /**
     * Construct a new Bias object.
     * @param value the severity of the bias.
     */
    public Bias(double value) 
    {
    	setValue(DEGREE,value);
    }
    Bias() 
    {}

	public Bias(String[] args)
	{
		setProperties(args);
	}

	public void setProperties(String[] args)
	{
		for (int x = 0; x < args.length; x+=2)
		{
			if(!args[x].startsWith("-") && args[x]!="")
				throw new IllegalArgumentException("Could not parse argument: " + args[x]);

			if (args[x]!="")
			{
				String name = args[x].substring(1);
				properties.put(name, args[x+1]);
			}
		}
	}

    /**
     * Inject this Bias into the given dataset and return the result.  The 
     * Dataset that is returned should always be a copy of the input Dataset,
     * rather than an in-place modification.
     * 
     * @param instances the Dataset in which to inject this Bias.
     * @param attribute the attribute (zero-based) along which to inject the
     * Bias.
     * @return a new, biased version of the input Dataset.
     */
    public abstract Dataset injectBias(Dataset instances, int attribute);
    /**
     * Return a string representing the whichUnit &quot;unit&quot; associated with
     * the severity of this bias.
     * @param the index of the severity measure
     * @return the measure for severity of this bias.
     */
    public abstract String getUnit(int whichUnit);
    
    /**
     * Returns the minimum acceptable value for the severity of this bias,
     * usually zero.  Used for user input validation
     * @return the minimum value.
     */
    public double getValueMin() { return 0.0; }
    /**
     * Returns the maximum acceptable value for the severity of this bias.
     * Used for user input validation
     * @return the maximum value.
     */
    public double getValueMax() { return 100.0; }
    /**
     * Sets the severity of this bias.
     * @param value the severity of this bias.
     */
    public void setValue(String key, double value)
    { 
    	properties.put(key, value); 
    }

    public double getValue()
	{
		return (Double)properties.get(DEGREE);
	}
    
	public double getDegree()
	{
		return (Double)properties.get(DEGREE);
	}
	
	public double getPHI()
	{
		return (Double)properties.get(PHI);
	}
	
	public double getGAMA()
	{
		return (Double)properties.get(GAMA);
	}
	
	public double getValue(String key)
	{
		return (Double.valueOf(properties.get(key).toString()));
	}

	public void setValue(double degree)
	{
		properties.put(DEGREE, degree);
	}
	
	public void setValue(double phi,double gama, double degree)
	{
		properties.put(DEGREE, degree);
		properties.put(PHI, phi);
		properties.put(GAMA, gama);
	}
	
	public boolean exists (String key)
	{
		return properties.containsKey(key);
	}
    

    /**
     * Return a string representation of this bias.
     * @return Currently, Name: value units.
     */
    public @Override String toString()
    {
    	String biasDescription = getName() + ": ";    	
    	return biasDescription;
    }

    /**
     * Gets the value of a named property.
     * @param name The name of a property defined with setProperty.
     * @return the value associated with the given property, or null if it is
     * undefined.
     */
    public Object getProperty(String name)
    {
        return properties.get(name);
    }

    /**
     * Sets the value of the given property to the given object.
     * @param name the property to define or redefine.
     * @param value the value to associate with the given property.
     */
    public void setProperty(String name, Object value)
    {
        properties.put(name, value);
    }
        
    public abstract String getName();
    public abstract String getLongDescription();
    
    public int getNumUnits() { return numUnits;}

	public String getUnits() { return "blah"; }

	
	/*
	 * Randomly choose parameters for the bias and inject it onto the original dataset.
	 * It overwrites the previous parameters the bias might have had.
	 * @return the dataset that results from injecting the bias onto original
	 */
	public abstract Dataset injectRandomBias (Dataset original, int attribute);

}
