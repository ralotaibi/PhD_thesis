/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package weka.ShiftInjection.basic;

/**
 * Abstract class representing the value of a single attribute of an instance
 * and its value.
 *
 * @see NumericAttribute
 * @see NominalAttribute
 * 
 * @author traeder
 */
public abstract class Attribute {
    
    /**
     * Constants defining whether the attribute is numeric or nominal.
     */
    public static enum AttributeType {NUMERIC, NOMINAL}
    
    protected AttributeType type;
    protected String[] possibleValues = null;
    protected String name;
    
    /**
     * Get the type (numeric or nominal) of the attribute.
     * @return the type.
     */
    public AttributeType getType() { return type; }
    
    /**
     * Set the value of this attribute as a string.
     * @param val the new value.
     */
    
    /**
     * Set the possible values for this attribute (for nominal attributes).
     * @param values array of possible attribute values.
     */
    public abstract void setPossibleValues(String[] values);
    /**
     * Get a list of the possible values for this attribute (for nominal attributes)
     * @return an array containing all possible attribute values.
     */
    public String[] getPossibleValues() { return possibleValues; }
    /**
     * Sets the name of this attribute,
     * @param name the name.
     */
    public void setName(String name) { this.name = name; }
    /**
     * Gets the name of this attribute.
     * @return the name.
     */
    public String getName() { return name; }
    /**
     * Determine whether this attribute is missing.  If so, the values returned
     * by getDoubleValue() etc. are invalid.
     * @return true if the attribute is missing, false otherwise.
     */
    
    /**
     * Convenience method for determining if an attribute is numeric.
     * @return true if the attribute is numeric
     */
    public abstract boolean isNumeric();
    /**
     * Convenience method for determining if the attribute is nominal.
     * @return true if the attribute is nominal.
     */
    public abstract boolean isNominal();

}
