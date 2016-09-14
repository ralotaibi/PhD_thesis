/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package weka.ShiftInjection.io;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Collection;
import java.util.TreeMap;
import weka.ShiftInjection.basic.Dataset;

/**
 *
 * @author traeder
 */
public abstract class FileFormat {
    
    private static final String[] defaultFormats = {"m2.io.C45FileFormat", "m2.io.ArffFileFormat"};
    private static final File FORMATS_FILE = new File(System.getProperty("user.home") + File.separator + ".m2" + File.separator + ".formats");
    private static TreeMap<String, FileFormat> formats = null;
    
    public static void readFormatFile()
    {
        String line;
        String key;
        FileFormat format;
        try {
            BufferedReader reader = new BufferedReader(new FileReader(FORMATS_FILE));
            while ((line = reader.readLine()) != null)
            {
                try {
                    format = (FileFormat)Class.forName(line).newInstance();
                    key = format.getShortName();
                    formats.put(key.toUpperCase(), format);
                } catch (Exception ex) {}
            }
        } catch (IOException e)
        {
            try {
                FORMATS_FILE.createNewFile();
            } catch (IOException ex) {}
        }
        
        for (String string : defaultFormats)
        {
            try {
                format = (FileFormat)Class.forName(string).newInstance();
                key = format.getShortName();
                formats.put(key.toUpperCase(), format);
            } catch (Exception ex) {}
        }
    }
    
    public static FileFormat getFormat(String shortName)
    {
        if (shortName.toUpperCase()=="TRA" || shortName.toUpperCase()=="TST")
        	return formats.get("ARFF");
        else	
        	return formats.get(shortName.toUpperCase());
    }
    
    public static Collection<String> getAvaliableFormats()
    {
        if (formats == null)
        {
            formats = new TreeMap();
            readFormatFile();
        }
        return formats.keySet();
    }
    
    /**
     * Create a DatasetReader that will properly read the given filename.
     * Which type of DatasetReader is returned is based on the file extension.
     * @param filename the filename to read.
     * @return a DatasetReader that will read the file. Defaults to C45Reader
     * @throws java.io.IOException if file reading throws an exception.
     */
    public static DatasetReader createDatasetReader(String filename) throws IOException
    {

        DatasetReader ret;
        if (formats == null)
        {
            formats = new TreeMap();
            readFormatFile();
        }
        for (String format : formats.keySet())
        {
            try {
                ret = formats.get(format).createReader(filename);
            } catch (Exception e) { e.printStackTrace(); ret = null; }
            if (ret != null)
                return ret;
        }
        return null;
    }
    
    /**
     * Create a DatasetWriter that will write the specified dataset to the
     * given file.  The type of DatasetWriter returned is based on the file
     * extension.
     * @param dataset the Dataset to write.
     * @param filename the file to write it to.
     * @return a DatasetWriter for doing the writing.  Defaults to C45Writer.
     * @throws java.io.IOException if file writing throws an exception.
     */
    public static DatasetWriter createDatasetWriter(Dataset dataset, String filename) throws IOException
    {
        DatasetWriter ret;
        if (formats == null)
        {
            formats = new TreeMap();
            readFormatFile();
        }
        for (String format : formats.keySet())
        {
            try {
                ret = formats.get(format).createWriter(dataset, filename);
            } catch (Exception e) { ret = null; }
            if (ret != null)
                return ret;
        }
        return null;
    }
    
    /**
     * Gets the default temporary-training-set filename for the given type of
     * file.
     * @param format Either IOFactory.C45_FORMAT or IOFactory.ARFF_FORMAT.
     * @return The name of the temporary file used to hold training sets.
     */
    public static String getDefaultTraining(FileFormat format)
    {
        return format.generateTempTrainingFile(null).getAbsolutePath();
    }
    
    /**
     * Gets the default temporary-test-set filename for the given type of file.
     * @param format Either IOFactory.C45_FORMAT or IOFactory.ARFF_FORMAT
     * @return The name of the temporary file used to hold test sets.
     */
    public static String getDefaultTesting(FileFormat format)
    {
        return format.generateTempTestingFile(null).getAbsolutePath();
    }
    
    public abstract DatasetReader createReader(String filename) throws IOException;
    public abstract DatasetWriter createWriter(Dataset dataset, String filename) throws IOException;
    public abstract File generateTempTrainingFile(String stem);
    public abstract File generateTempTestingFile(String stem);
    public String getShortName() { throw new UnsupportedOperationException("FileFormat subclass needs short name.\n"); }

}
