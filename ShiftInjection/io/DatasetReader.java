/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package weka.ShiftInjection.io;

import weka.ShiftInjection.basic.Dataset;
import java.io.IOException;

/**
 * Interface implemented by any class capable prodducing a Dataset object from
 * input, usually a file.
 * 
 * @author traeder
 */
public abstract class DatasetReader {
    
    /**
     * Read a Dataset from its source.
     * @return the Dataset read.
     * @throws java.io.IOException if I/O accidents warrant.
     */
    public abstract Dataset read() throws IOException;

}
