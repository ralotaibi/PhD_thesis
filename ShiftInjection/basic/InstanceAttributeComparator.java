/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package weka.ShiftInjection.basic;

import java.util.Comparator;

/**
 * Comparator subclass used for sorting a dataset by an attribute.
 * 
 * @author traeder
 */
public class InstanceAttributeComparator implements Comparator {
    
    private int index;
   
    /**
     * Make a new InstanceAttributeComparator.
     * @param index the index of the attribute to sort by.
     */
    public InstanceAttributeComparator(int index)
    {
        this.index = index;
    }
    
    /**
     * Comparison function to compare two attributes for sorting.  The rules are:
     * 
     * <ul><li>Attempting to compare numeric and nominal attributes results in an
     * Exception</li>
     * <li>Numeric attributes are compared based on their values.</li>
     * <li>Nominal attributes are compared based on their String values.</li>
     * <li>Missing values are equal to other missing values and greater than non-
     * missing values.</li></ul>
     * 
     * @param o1 Attribute object representing the first attribute.
     * @param o2 Attribute object representing the second attribute.
     * @return 1 if o1 &gt; o2, 0 if they are equal, and -1 otherwise.
     * @throws IllegalArgumentException, if the two Attributes are of different
     * types.
     * @throws ClassCastException if something other than an Attribute is
     * passed in.
     */
    public int compare(Object o1, Object o2)
    {
        Attribute a1 = ((Instance) o1).getDataset().getAttribute(index);
        Attribute a2 = ((Instance) o2).getDataset().getAttribute(index);
        
        if (a1.getType() != a2.getType())
            throw new IllegalArgumentException("Attempting to compare attributes of different types!");
        if (((Instance)o1).isMissing(index))
            if (((Instance)o2).isMissing(index))
                return 0;
            else
                return 1;
        else if (((Instance)o2).isMissing(index))
            return -1;
        else if (a1.getType() == Attribute.AttributeType.NUMERIC)
            return Double.compare(((Instance)o1).doubleValue(index), ((Instance)o2).doubleValue(index));
        else
            return Double.compare(((Instance)o1).intValue(index), ((Instance)o2).intValue(index));
    }

}
