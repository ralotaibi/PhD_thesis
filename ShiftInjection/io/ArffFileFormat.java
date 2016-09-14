package weka.ShiftInjection.io;

import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import weka.ShiftInjection.basic.Dataset;

public class ArffFileFormat extends FileFormat
{

    public @Override String getShortName() 
    {
        return "ARFF";
    }

    public DatasetReader createReader(String filename) throws IOException 
    {
        if (filename.endsWith(".arff")) 
            return new ArffReader(new FileReader(new File(filename).getAbsolutePath()));
        return null;
    }

    public DatasetWriter createWriter(Dataset dataset, String filename) throws IOException 
    {
        if (filename.endsWith(".arff")) 
         return new ArffWriter(dataset, new FileWriter(filename));
        return null;
    }
    
    public File generateTempTrainingFile(String stem)
    {
       if (stem == null)
           return new File(System.getProperty("user.home") + File.separator + ".m2" + File.separator + "temp_train.arff");
       else
           return new File(stem + "_train.arff");
    }
    
    public File generateTempTestingFile(String stem)
    {
       if (stem == null)
           return new File(System.getProperty("user.home") + File.separator + ".m2" + File.separator + "temp_test.arff");
       else
           return new File(stem + "_test.arff");
    }
}
