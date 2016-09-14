/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package weka.ShiftInjection.io;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

import weka.ShiftInjection.basic.Dataset;
import weka.ShiftInjection.basic.Attribute;
import java.io.PrintWriter;
import java.io.Writer;

/**
 * DatasetWriter for writing a Dataset to a Writer in ARFF format.
 * @author traeder
 */
public class ArffWriter extends DatasetWriter{
    
    private PrintWriter writer;
    private Dataset dataset;

    /**
     * Construct an ArffWriter that will write the given dataset to the given
     * Writer in ARFF format.
     * @param d a Dataset to write.
     * @param w a Writer to write it to.
     */
    public ArffWriter(Dataset d, Writer w)
    {        
        dataset = d;
        writer = new PrintWriter(w);
    }
    
    /**
     * Write the Dataset to the Writer in ARFF format.
     */
    public void write()
    {
        Attribute attribute;
        String name=dataset.getName();
        name="@relation 'DataSet: -C -1'";
        writer.write(name+"\n");
        String[] values;
        String tempy;
        
        for (int x = 0; x < dataset.numAttributes(); x++)
        {
            attribute = dataset.getAttribute(x);
            writer.write("@attribute " + attribute.getName());
            if (attribute.getType() == Attribute.AttributeType.NUMERIC)
                 writer.write(" numeric\n");
            else
            {
                values = attribute.getPossibleValues();
                tempy = " {" + values[0];
                for (int y = 1; y < values.length; y++)
                    tempy += ("," + values[y]);
                tempy += "}\n";
                writer.write(tempy);
            }
        }
            
        writer.write("@data\n");

        for (int x = 0; x < dataset.numInstances(); x++)
        {
            writer.write(dataset.getInstance(x).toString());
            writer.write("\n");
        }
        
        writer.close();
    }
}
