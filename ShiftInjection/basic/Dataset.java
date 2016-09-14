
package weka.ShiftInjection.basic;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;
import weka.ShiftInjection.basic.myRandomizer;


public class Dataset 
{
    
    private ArrayList<Attribute> attributes = new ArrayList();
    private ArrayList<Instance> instances = new ArrayList();
    private String name;
    
    /**
     * Add an attribute to this dataset.
     * @param attribute the attribute to add.
     */
    public void addAttribute(Attribute attribute)
    {
        attributes.add(attribute);
    }
    
    /**
     * Returns the attribute at the given index from the dataset.  These
     * attributes will not have values, so it's only useful to get names
     * and types.
     * @param index the index of the attribute to retrieve.
     * @return the Attribute.
     */
    public Attribute getAttribute(int index)
    {
        return attributes.get(index);
    }
    
    /**
     * Sets the name of this Dataset, for friendly display purposes.
     * @param name a friendly, human-readable name.
     */
    public void setName(String name)
    {
        this.name = name;
    }
    
    /**
     * Gets the name of this Dataset.
     * @return a friendly human-readable name.
     */
    public String getName()
    {
        return name;
    }
    
    /**
     * Creates and returns a new dataset that has the exact same attributes
     * of this dataset but has no instances.
     * 
     * @return the empty dataset.
     */
    public Dataset emptyClone()
    {
        Dataset ret = new Dataset();
        
        for (Attribute attribute : attributes)
            if (attribute.getType() == Attribute.AttributeType.NUMERIC)
                ret.addAttribute(new NumericAttribute(attribute));
            else
                ret.addAttribute(new NominalAttribute(attribute));

        ret.setName(name);
        return ret;
    }
    
    /**
     * Does a deep clone of this dataset and returns the result.
     * @return an exact copy of the current dataset.
     */
    
    public @Override Dataset clone()
    {
        Dataset ret = emptyClone();
        
        for (Instance instance : instances)
            ret.addInstanceClone(instance);
        
        return ret;
    }
    
    /**
     * Creates and returns an instance that has all the attributes of the
     * current dataset, but whose attributes have no values.
     * @return the empty instance.
     */
    public Instance emptyInstance()
    {
        Instance instance = new Instance(attributes.size());
        instance.setDataset(this);
        return instance;
    }
     
    /**
     * Convenience method to create an instance from a comma-separated string
     * and add it do this dataset.  The line must conform to the schema of this
     * dataset, otherwise the method will fail.
     * 
     * @param line a comma-separated string representing the instance to add.
     */
    public void addInstance(String line)
    {
        Instance instance = emptyInstance();
        instance.initialize(line);
        instance.setDataset(this);
        instances.add(instance);
    }
    
    /**
     * Adds an instance to a dataset without copying it.  Only the reference
     * is added, meaning that if it is changed later, the dataset will change.
     * To add a copy of the instance use addInstanceClone()
     * 
     * @param instance the Instance to add.
     */
    public void addInstance(Instance instance)
    {
        instances.add(instance);
    }
    
    /**
     * Creates a deep copy of the given instance and adds it to the current
     * dataset.  The instance must conform to the schema of this dataset.
     * 
     * @param instance the instance to add.
     */
    public void addInstanceClone(Instance instance)
    {
        instances.add(instance.clone());
    }
    
    
    /**
     * Returns the instance at the given index of the current dataset.
     * @param x tje instance of the index to return.
     * @return the appropriate Instance
     */
    public Instance getInstance(int x) { return instances.get(x); }
    
    /**
     * @return the number of instances in this dataset.
     */
    public int numInstances() { return instances.size(); }
    
    /**
     * @return the number of attributes in this dataset.
     */
    public int numAttributes() { return attributes.size(); }
    
    /**
     * Sorts this dataset by the given attribute, in ascending order.
     * @param x the zero-based index of the attribute on which to sort.
     */
    public void sortByAttribute(int x)
    {
        Collections.sort(instances, new InstanceAttributeComparator(x));
    }
    
    /**
     * Returns the mean of the given attribute, assuming it is a numeric
     * attribute.
     * @param attribute the zero-based index of the attribute.
     * @return the mean of the give attribute.
     */
    public double getMean(int attribute)
    {
        double ret = 0;
        int count = 0;
        
        for (Instance instance : instances)
        {
            if (!instance.isMissing(attribute))
            {
                ret += instance.doubleValue(attribute);
                count++;
            }
        }
        
        return ret / count;
    }
    /**
     * Creates a new dataset by bootstrap sampling of this dataset.
     * @param percentage percentage of the instances to sample.  Instances are
     * sampled with replacement.
     * @return the new dataset.
     */
    public Dataset bootstrap(double percentage)
    {
        int size = (int) (percentage  * numInstances() / 100);
        Dataset ret = emptyClone();
        Random rand = new Random(System.currentTimeMillis());
        int selection;
        
        for (int x = 0; x < size; x++)
        {
            selection = rand.nextInt(numInstances());
            ret.addInstanceClone(instances.get(selection));
        }
        
        return ret;
    }
    
    /**
     * Gets the value of an attribute for all instances in the dataset as an int
     * array.
     * @param attribute the zero-based attribute index.
     * @return the array of values.
     */
    public int[] getIntValues(int attribute)
    {
        int[] ret = new int[numInstances()];
        
        for (int x = 0; x < numInstances(); x++)
            ret[x] = getInstance(x).intValue(attribute);
        
        return ret;
    }
    
    /**
     * Shuffles this dataset so its instances are in a random order.
     */
    public void randomize()
    {
        Collections.shuffle(instances);
    }
    
    /**
     * @param attribute the zero-based index of an attribute.
     * @return the variance of the given attribute, assuming it is numeric.
     */   
    public double getVariance(int attribute)
    {
        double mean = getMean(attribute);
        double ret = 0;
        int count = 0;
        
        for (Instance instance : instances)
        {
            if (!instance.isMissing(attribute))
            {
                ret += Math.pow(instance.doubleValue(attribute) - mean, 2);
                count++;
            }
        }
        
        return ret / (count - 1);
    }
    
    /**
     * @param attribute the zero-based index of an attribute.
     * @return the minimum value of the given attribute assuming it is numeric.
     */
    public double getMin(int attribute)
    {
        double min = Double.MAX_VALUE;
        
        for (Instance instance : instances)
            if (!instance.isMissing(attribute) && min > instance.doubleValue(attribute))
                min = instance.doubleValue(attribute);
        return min;
    }
    
    /**
     * @param attribute the zero-based index of an attribute.
     * @return the minimum value of the given attribute assuming it is numeric.
     */
    public int getRandomPoint(int attribute)
    {
    	boolean flag=false;
    	int index=0;
    	System.out.println("instances size="+instances.size());
    	while (flag!=true)
    	{
        index=myRandomizer.generator.nextInt();
        if (index>=0 && index <instances.size())
        	{
        	flag=true;
        	}

    	}
        return index;
    }
    
    /**
     * @param attribute the zero-based index of an attribute.
     * @return the maximum value of an attribute, assuming it is numeric.
     */
    public double getMax(int attribute)
    {
        double max = Double.MIN_VALUE;
        
        for (Instance instance : instances)
            if (!instance.isMissing(attribute) && max < instance.doubleValue(attribute))
                max = instance.doubleValue(attribute);
        return max;
    }
    
    /**
     * @param attribute the zero-based index of an attribute.
     * @return an array of doubles containing the value of the given attribute
     * for all instances in the dataset.  Instances with missing values are
     * omitted.
     */
    public double[] todoubleArray(int attribute)
    {
        double[] ret = new double[numInstances()];
        double[] finalArr;
        int x = 0;
        
        for (Instance instance : instances)
        {
            if (!instance.isMissing(attribute))
            {
                ret[x] = instance.doubleValue(attribute);
                x++;
            }
        }
        
        finalArr = new double[x];
        System.arraycopy(ret, 0, finalArr, 0, x);
        return finalArr;
    }
    
     /**
     * @param attribute the zero-based index of an attribute.
     * @return an array of Double objects containing the value of the given
     * attribute for all instances in the dataset.  Instances with missing 
     * values are omitted.
     */
    public Double[] toDoubleArray(int attribute)
    {
        Double[] ret = new Double[numInstances()];
        Double[] finalArr;
        int x = 0;
        
        for (Instance instance : instances)
        {
            if (!instance.isMissing(attribute))
            {
                ret[x] = instance.doubleValue(attribute);
                x++;
            }
        }
        
        finalArr = new Double[x];
        System.arraycopy(ret, 0, finalArr, 0, x);
        return finalArr;
    }
    
    /**
     * Removes the given attribute from the dataset and all its instances.
     * @param attribute the index of the attribute to remove.
     */
    public void removeAttribute(int attribute)
    {
        attributes.remove(attribute);
        
        for (Instance instance : instances)
            instance.removeAttribute(attribute);
    }
    
    public void removeInstance(int instanceIndex){
    	instances.remove(instanceIndex);
    }
    
    public double[] toConditionaldoubleArray(int attribute, int classValue)
    {
        ArrayList<Double> array = new ArrayList();
        double[] ret;
        
        if (classValue < 0)
            return todoubleArray(attribute);
        
        for (int x = 0; x < numInstances(); x++)
            if (instances.get(x).intValue(this.numAttributes() - 1) == classValue)
                array.add(instances.get(x).doubleValue(attribute));
        
        ret = new double[array.size()];
        for (int x = 0; x < array.size(); x++)
            ret[x] = array.get(x);
        
        return ret;

    }
    
    public Double[] toConditionalDoubleArray(int attribute, int classValue)
    {
        ArrayList<Double> array = new ArrayList();
        
        if (classValue < 0)
            return toDoubleArray(attribute);
        
        for (int x = 0; x < numInstances(); x++)
            if (instances.get(x).intValue(this.numAttributes() - 1) == classValue)
                array.add(instances.get(x).doubleValue(attribute));
        
        return array.toArray(new Double[0]);
    }
    
    public int[] toConditionalIntArray(int attribute, int classValue)
    {
        ArrayList<Integer> array = new ArrayList();
        int[] ret;
        
        if (classValue < 0)
            return getIntValues(attribute);
        
        for (int x = 0; x < numInstances(); x++)
            if (instances.get(x).intValue(this.numAttributes() - 1) == classValue)
                array.add(instances.get(x).intValue(attribute));
        
        ret = new int[array.size()];
        for (int x = 0; x < array.size(); x++)
            ret[x] = array.get(x);
        
        return ret;

    }
    
    public double getConditionalMin(int attribute, int classIndex)
    {
        double min = Double.POSITIVE_INFINITY;
        
        if (classIndex < 0)
            return getMin(attribute);
        
        for (Instance instance : instances)
            if (!instance.isMissing(attribute) && instance.intValue(attributes.size() - 1) == classIndex && instance.doubleValue(attribute) < min)
                min = instance.doubleValue(attribute);
        return min;
    }
    
    public double getConditionalMax(int attribute, int classIndex)
    {
        double max = Double.NEGATIVE_INFINITY;
        
        if (classIndex < 0)
            return getMax(attribute);
        
        for (Instance instance : instances)
            if (!instance.isMissing(attribute) && instance.intValue(attributes.size() - 1) == classIndex && instance.doubleValue(attribute) > max)
                max = instance.doubleValue(attribute);
        return max;
    }
    
    public double getConditionalMean(int attribute, int classIndex)
    {
        double sum = 0;
        int count = 0;
        
        if (classIndex < 0)
            return getMean(attribute);
        
        for (Instance instance : instances)
        {
            if (!instance.isMissing(attribute) && instance.intValue(attributes.size() - 1) == classIndex)
            {
                count++;
                sum += instance.doubleValue(attribute);
            }
        }
        return sum / count;
    }
    
    public double getConditionalVariance(int attribute, int classIndex)
    {
        double mean;
        double ret = 0;
        int count = 0;
        
        if (classIndex < 0)
            return getVariance(attribute);
        
        mean = getConditionalMean(attribute, classIndex);
        
        for (Instance instance : instances)
        {
            if (!instance.isMissing(attribute))
            {
                ret += Math.pow(instance.doubleValue(attribute) - mean, 2);
                count++;
            }
        }
        
        return ret / (count - 1);
    }
    
    public int getConditionalNumInstances(int classIndex)
    {
        int count = 0;
        
        if (classIndex < 0)
            return numInstances();
        
        for (Instance instance : instances)
            if (instance.intValue(instance.numAttributes() - 1) == classIndex)
                count++;
        return count;
    }
    
    public void trim()
    {
        attributes.trimToSize();
        instances.trimToSize();
    }
    
    public Dataset split(int i, int j)
    {
    	Dataset ret = emptyClone();
    	int index;
    	for (index=i; index<=j;index++)
    	if(index<instances.size())
    		ret.addInstance(instances.get(index));
    	return ret;
    }
    
    public Dataset firstHalf(){
    	Dataset ret = emptyClone();
    	for (int i=0; i<instances.size()/2;i++)
    		ret.addInstance(instances.get(i));
    	return ret;
    }
    
    public Dataset secondHalf(){
    	Dataset ret = emptyClone();
    	for (int i=instances.size()/2;i<instances.size();i++)
    		ret.addInstance(instances.get(i));
    	return ret;
    }
    
    public Dataset[] getTrainingCV(int k){
    	Dataset[] ret = new Dataset[k];
    	for (int f=0; f<k; f++){
    		ret[f] = emptyClone();
    	}
    	for (int inst=0;inst<numInstances();inst++)
    		for (int f=0;f<k;f++)
    			if (inst%k!=f)
    				ret[f].addInstanceClone(instances.get(inst));
    		
		return ret;
    }
    
    public Dataset[] getTestCV(int k){
    	Dataset[] ret = new Dataset[k];
    	for (int f=0; f<k; f++){
    		ret[f] = emptyClone();
    	}
    	for (int inst=0;inst<numInstances();inst++)
    		for (int f=0;f<k;f++)
    			if (inst%k==f)
    				ret[f].addInstanceClone(instances.get(inst));
    		
		return ret;
    }
    
    
    public void createPartitions(){
    	Dataset copy = clone();
    	boolean[] used = new boolean[numInstances()];
    	for (int i=1; i<numInstances(); i++)
    		used[i]=false;
    	
    	Instance previous = copy.instances.get(0);
    	int i;
    	for (i=1; i<numInstances(); i++){
    		copy.instances.set(i, nextElement(used, previous));
    		previous = copy.instances.get(i);
    	}
    	
    	instances = (ArrayList<Instance>) copy.instances.clone();
    }
    
	private Instance nextElement(boolean[] used, Instance previous){
		double distance = Double.MAX_VALUE;
		Instance currentWinner = null;
		int winnerIndex=0;
		for (int i=1; i<numInstances()-1;i++)
			if (previous.myGetClass()==instances.get(i).myGetClass() && !used[i] &&previous.Distance(instances.get(i))<distance){
				currentWinner = instances.get(i).clone();
				distance = previous.Distance(currentWinner);
				winnerIndex=i;
			}
		
		if (winnerIndex==0){
			for (int i=1; i<numInstances();i++)
				if (!used[i] &&previous.Distance(instances.get(i))<distance){
					currentWinner = instances.get(i).clone();
					distance = previous.Distance(currentWinner);
					winnerIndex=i;
				}
			//System.out.print("crap");
		}
		
		for (int i=1; i<numInstances()-1;i++)
			if (previous.myGetClass()==instances.get(i).myGetClass() && !used[i]){
				currentWinner = instances.get(i).clone();
				winnerIndex=i;
			}
		
		if (winnerIndex==0){
			for (int i=1; i<numInstances();i++)
				if (!used[i]){
					currentWinner = instances.get(i).clone();
					winnerIndex=i;
				}
			//System.out.print("crap");
		}
		used[winnerIndex]=true;
		if (currentWinner==null)
			System.out.print("crap");
		return currentWinner;
	}
}
