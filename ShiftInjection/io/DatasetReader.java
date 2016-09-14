
package weka.ShiftInjection.io;

import weka.ShiftInjection.basic.Dataset;
import java.io.IOException;


public abstract class DatasetReader {
    
    /**
     * Read a Dataset from its source.
     * @return the Dataset read.
     * @throws java.io.IOException if I/O accidents warrant.
     */
    public abstract Dataset read() throws IOException;

}
