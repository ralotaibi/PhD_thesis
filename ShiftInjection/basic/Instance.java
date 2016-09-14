
package weka.ShiftInjection.basic;


public class Instance 
{
    
    private double[] values;
    private int numAttributes = 0;
    private Dataset dataset;
    
   /**
    * Construct an empty instance with the given number of attributes.
    * @param attributes the number of attributes for the new instance.
    */
   public Instance(int numAttributes)
   {
       values = new double[numAttributes];
       this.numAttributes = numAttributes;
   }
   
   public Dataset getDataset()
   {
       return dataset;
   }
   
   public void setDataset(Dataset dataset)
   {
       this.dataset = dataset;
   }
    
   /**
    * Create an empty clone of the given set of attributes (same names and
    * types, but no values).
    * 
    * @param attributes a list of attributes to clone.
    */
    /*public void cloneAttributes(ArrayList<Attribute> attributes)
    {
        this.attributes.clear();
        for (Attribute attribute : attributes)
            if (attribute.getType() == Attribute.AttributeType.NUMERIC)
                this.attributes.add(new NumericAttribute(attribute));
            else
                this.attributes.add(new NominalAttribute(attribute));
        attributes.trimToSize();
    }*/
    
    /**
     * Initialize an instance from a comma-separated line in an input file.
     * 
     * @param line a comma-separated list of attribute values that fits with this
     * Instance's set of attributes.
     */
    public void initialize(String line)
    {
        String[] arr;
        
        line = line.replaceAll(",\\s+", ",");
        arr = line.split(",");
        
        for (int x = 0; x < arr.length; x++)
            setAttributeValue(x, arr[x]);
    }    
    
    
    public void setAttributeValue(int attribute, String value)
    {
        String[] possiblevalues;
        
        if (value.equals("?"))
            values[attribute] = Double.NaN;
        else if (dataset.getAttribute(attribute).isNumeric())
            values[attribute] = Double.valueOf(value);
        else
        {
            values[attribute] = -1;
            possiblevalues = dataset.getAttribute(attribute).getPossibleValues();
            for (int x = 0; x < possiblevalues.length; x++)
                if (value.toLowerCase().trim().replaceAll("\"", "").equals(possiblevalues[x].toLowerCase().trim().replaceAll("\"", "")))
                    values[attribute] = x;
            if (values[attribute] == -1)
//                throw new IllegalArgumentException("Could not assign value \"" + value + "\" to attribute " + dataset.getAttribute(attribute).getName());
            	values[attribute]=Integer.valueOf(value);
        }
    }
    
    public void setAttributeValue(int attribute, double value)
    {
        values[attribute] = value;
    }
    
    public void setMissing(int attribute)
    {
        values[attribute] = Double.NaN;
    }
    
    /**
     * Clones this instance (attributes and values)
     * @return an exact copy of this Instance.
     */
    public @Override Instance clone()
    {
        Instance ret = new Instance(values.length);
        ret.setDataset(dataset);
        
        for (int x = 0; x < values.length; x++)
            ret.values[x] = values[x];
        
        return ret;
    }
    
    /**
     * Returns a string represenation of this instance.
     * @return A comma-separated list of current attribute values.
     */
    public @Override String toString()
    {
        String ret = "";
        int x;
        for (x = 0; x < numAttributes - 1; x++)
            ret += stringValue(x) + ",";
        
        ret += stringValue(x);
        
        return ret;
    }
    
    /**
     * Get the number of attributes this instance has.
     * @return the number of attributes.
     */
    public int numAttributes()
    {
        return numAttributes;
    }
    
    /**
     * returns the attribute at the given index in this Instance's list of
     * attributes.
     * @param x the index.
     * @return the Attribute in question.
     */
    //public Attribute getAttribute(int x) { return attributes.get(x); }
    
    /**
     * Removes the Attribute at the given index from this Instance's list of
     * attributes.
     * @param x the index.
     */
    public void removeAttribute(int x) 
    {
        numAttributes--;
        for (int y = x; y < numAttributes; y++)
            values[y] = values[y + 1];
    }
    
    public int intValue(int x)
    {
        return (int)values[x];
    }
    
    public double doubleValue(int x)
    {
        return values[x];
    }
    
    public String stringValue(int x)
    {
        if (isMissing(x))
            return "?";
        else if (dataset.getAttribute(x).isNominal())
            return dataset.getAttribute(x).getPossibleValues()[(int)values[x]];
        else
            return String.valueOf(values[x]);
    }
    
    public boolean isMissing(int x)
    {
        return Double.isNaN(values[x]);
    }
    
    public double Distance (Instance other){
    	double distance=0;
    	for (int at=0; at<numAttributes(); at++) {
    		if (dataset.getAttribute(at).isNumeric())
    			distance+=Math.abs(other.doubleValue(at)-doubleValue(at));
    		else
    			if (other.doubleValue(at)!=doubleValue(at))
    				distance+=1/dataset.getAttribute(at).getPossibleValues().length;
    	}
    		
    	return distance;
    }
    
    public int myGetClass(){
    	return (int) values[numAttributes-1];
    }

}
