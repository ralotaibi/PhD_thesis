/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package weka.ShiftInjection.io;

import java.io.IOException;
import weka.ShiftInjection.basic.Dataset;

/**
 * Interface implemented by any class capable of writing a Dataset object in
 * some meaningful way, usually to a file.
 * 
 * @author traeder
 */
public abstract class DatasetWriter 
{

    /**
     * Write the dataset.
     * @throws java.io.IOException if I/O accidents warrant.
     */
    public abstract void write() throws IOException;
    
}
