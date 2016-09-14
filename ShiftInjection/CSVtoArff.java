package weka.ShiftInjection;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;

public class CSVtoArff 
{

	public static void main(String[] args) throws IOException 
	{
		String folder = "/Users/ra12404/Desktop/meka-1.7.3/data/Bike/Challenge/stations_challenge_tra/";
		String targetstations="stations_challenge_75";
		//String histstations="stations_challenge_200";
		String trFileName;

		for (int f=1; f<276;f++)
		{			    
			trFileName = folder+targetstations+"/station_"+f+"_challenge"+".arff";
			File file=new File(trFileName);
			if(file.exists())
			{
		    RandomAccessFile fo = new RandomAccessFile(file, "rw");
		    fo.seek(0); // to the beginning
		    fo.write("@relation 'Data: -C -1'".getBytes());
		    fo.write("\n".getBytes());

		    fo.close();
			}
		}

	}

}
