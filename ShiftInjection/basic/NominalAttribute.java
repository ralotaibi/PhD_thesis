/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package weka.ShiftInjection.basic;

/**
 * Attribute subclass for holding nominal attributes.
 * 
 * @author traeder
 */
public class NominalAttribute extends Attribute {
    
    public NominalAttribute(String name) { this.name = name; type = AttributeType.NOMINAL; }
    public NominalAttribute(String name, String[] possibleValues)
    {
        this.name = name;
        type = AttributeType.NOMINAL;
        this.setPossibleValues(possibleValues);
    }
    public NominalAttribute(Attribute template)
    {
        this.name = template.name;
        this.possibleValues = template.possibleValues;
    }
    
    /**
     * Sets the list of possible values for this attribute.
     * @param values an array holding all the possible values of this attribute.
     */
    public void setPossibleValues(String[] values) 
    {
        possibleValues = new String[values.length];
        for (int x = 0; x < values.length; x++)
            possibleValues[x] = values[x].trim();
    }
    
    /**
     * @return false
     */
    public boolean isNumeric() { return false; }
    /**
     * @return true
     */
    public boolean isNominal() { return true; }
}
