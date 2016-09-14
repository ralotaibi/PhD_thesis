/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package weka.ShiftInjection.basic;

/**
 * Attribute subclass for representing numeric attributes.
 * @author traeder
 */
public class NumericAttribute extends Attribute {
    
    /**
     * Construct a new numeric attribute.
     * @param name a unique name for the attribute.
     */
    public NumericAttribute(String name) {this.name = name; this.type = AttributeType.NUMERIC;}
    
    /**
     * Constructs an empty clone of the given attribute (same name, but no value).
     * @param template the attribute to clone.
     */
    public NumericAttribute(Attribute template)
    {
        this.name = template.name;
        this.type = AttributeType.NUMERIC;
    }
    
    /**
     * Throws an UnsupportedOperationException
     * @param values it doesn't matter.
     */
    public void setPossibleValues(String[] values) { throw new UnsupportedOperationException("Cannot set possible values for Numeric attributes."); }
    
    /**
     * @return true
     */
    public boolean isNumeric() { return true; }
    /**
     * @return false
     */
    public boolean isNominal() { return false; }
    
}
