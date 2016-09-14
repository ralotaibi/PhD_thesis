
package weka.ShiftInjection.io;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.Reader;
import java.io.StreamTokenizer;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import weka.ShiftInjection.basic.Dataset;
import weka.ShiftInjection.basic.NominalAttribute;
import weka.ShiftInjection.basic.NumericAttribute;


public class ArffReader extends DatasetReader {
    
    private BufferedReader reader;
    private StreamTokenizer m_Tokenizer;
    
        /**
     * Initializes the StreamTokenizer used for reading the ARFF file.
     */
    protected void initTokenizer(){
      m_Tokenizer.resetSyntax();         
      m_Tokenizer.whitespaceChars(0, ' ');    
      m_Tokenizer.wordChars(' '+1,'\u00FF');
      m_Tokenizer.whitespaceChars(',',',');
      m_Tokenizer.commentChar('%');
      m_Tokenizer.quoteChar('"');
      m_Tokenizer.quoteChar('\'');
      m_Tokenizer.ordinaryChar('{');
      m_Tokenizer.ordinaryChar('}');
      m_Tokenizer.eolIsSignificant(true);
    }
    
    /**
     * Build a new ArffReader to read from the underlying Reader.
     * @param reader any Reader object containing a representation of an ARFF
     * file.
     */
    public ArffReader(Reader reader)
    {
        this.reader = new BufferedReader(reader);
    }
    
        
    /**
     * Read from the given reader and return a Dataset object representing
     * the data it contains.
     * @return a Dataset object containing all the data from the Reader.
     * @throws java.io.IOException if the ARFF file is misformatted.
     */
    public Dataset read() throws IOException
    {
        String line;
        String line2;
        Dataset ret = new Dataset();
        String numericMatch = "@attribute\\s+(.*)\\s+numeric(\\s*\\[(.*?)\\])?+$";
//        String numericMatch = "@attribute\\s+(.*)\\s+integer(\\s*\\[(.*?)\\])?+$";
        String realMatch = "@attribute\\s+(.*)\\s+real(\\s*\\[(.*?)\\])?+";
        String nominalMatch = "@attribute\\s+(.*?)\\s*\\{(.*?)\\}";
        Pattern numericPattern = Pattern.compile(numericMatch, Pattern.CASE_INSENSITIVE);
        Pattern nominalPattern = Pattern.compile(nominalMatch, Pattern.CASE_INSENSITIVE);
        Pattern realPattern = Pattern.compile(realMatch, Pattern.CASE_INSENSITIVE);
        Matcher matcher;
        String name, possibleValues;
        ArrayList<String> tempPossibleValues;
                        
        while ((line = reader.readLine()) != null && !line.equalsIgnoreCase("@data"))
        {
            line = line.toLowerCase().trim();
            if (line.indexOf("@relation") >= 0)
                ret.setName(line.substring(line.lastIndexOf("+") + 1));
            if (line.indexOf("@attribute") >= 0)
            {
                line2 = line.replaceAll("\\s", "");
                if (line2.length() > 0 && !line2.startsWith("%"))
                {
                    matcher = numericPattern.matcher(line);
                    if (matcher.matches())
                        ret.addAttribute(new NumericAttribute(matcher.group(1)));
                    else
                    {
                        matcher = realPattern.matcher(line);
                        if (matcher.matches())
                            ret.addAttribute(new NumericAttribute(matcher.group(1)));
                        else
                        {
                            matcher = nominalPattern.matcher(line);
                            if (matcher.matches())
                            {
                                name = matcher.group(1);
                                possibleValues = matcher.group(2);
                                m_Tokenizer = new StreamTokenizer(new StringReader(possibleValues));
                                initTokenizer();
                                tempPossibleValues = new ArrayList();
                                while (m_Tokenizer.nextToken() != StreamTokenizer.TT_EOF)
                                    tempPossibleValues.add(m_Tokenizer.sval);
                                ret.addAttribute(new NominalAttribute(name, tempPossibleValues.toArray(new String[0])));
                            }
                            else
                                throw new IOException("Could not understand attribute line: " + line);
                        }
                    }
                }
            }
        } // end reading @attributes
        if (line == null) throw new IOException("Error: No @data line found!");
        
        while ((line = reader.readLine()) != null)
            if (line.length() > 0 && !line.startsWith("%"))
                ret.addInstance(line);
        ret.trim();
        
        return ret;
    }
}
