package weka.ShiftInjection.bias;

import java.util.Arrays;
import weka.ShiftInjection.basic.Dataset;
import weka.ShiftInjection.basic.Instance;



public class PicewiseNonLinearShift extends Bias
{
	 
    public PicewiseNonLinearShift() 
    { 
    	super(); 
    }   
    /**
     * Minimum allowable value for the number of standard deviations.
     * @return Double.NEGATIVE_INFINITY
     */
    public @Override double getValueMin()
    {
        return Double.NEGATIVE_INFINITY;
    }   
    /**
     * Maximum allowable value for number of standard deviations
     * @return Double.POSITIVE_INFINITY
     */
    public @Override double getValueMax()
    {
        return Double.POSITIVE_INFINITY;
    }   
    public PicewiseNonLinearShift(double value)
    {
        super(value);     
    } 
    
	@Override
    public Dataset injectBias(Dataset dataset, int attribute)
    {
    	int []cuts=new int [4];
    	int range=dataset.numInstances()/4;
    	for(int i=0;i<4;i++)
    	{
    		cuts[i]=((i+1)*range)-1;
    		System.out.println("c="+cuts[i]);
    	}
		
		Dataset [] splits=new Dataset[4];
		double [] thresholds=new double[4];
		double []means=new double[4];
		double []sds=new double[4];
		double globalmean=dataset.getMean(attribute);
		double globalsd=Math.sqrt(dataset.getVariance(attribute));
		System.out.println("global mean="+globalmean+" and sd="+globalsd);
		
		int inx=0;
		Arrays.sort(cuts);
		dataset.sortByAttribute(attribute);
		
		splits[0]=dataset.split(0, cuts[0]);
		splits[1]=dataset.split(cuts[0], cuts[1]);
		splits[2]=dataset.split(cuts[1], cuts[2]);
		splits[3]=dataset.split(cuts[2], cuts[3]);

		for(int x = 0; x < cuts.length; x++)
		{
			means[x]=splits[x].getMean(attribute);
			sds[x]=Math.sqrt(splits[x].getVariance(attribute));
		}
		
		for(int x = 0; x < cuts.length; x++)
		{
			thresholds[x]=dataset.getInstance(cuts[x]).doubleValue(attribute);
		}
		
		double attvalue=0,mult,add;
		Instance instance;
        Dataset copy = dataset.clone();
        

		 for (int x = 0; x < dataset.numInstances(); x++)
         {
			 attvalue = copy.getInstance(x).doubleValue(attribute);
			 instance = copy.getInstance(x);
			 			
			  if  (attvalue<thresholds[0])					
			 {
                 mult=Math.pow(2, -1);
         		 add=(1-Math.pow(2, -1))*globalmean+(1*globalsd);
      			// System.out.println("mult="+mult+" and add="+add);
      			// System.out.println("instance before="+instance.doubleValue(attribute));

                 copy.getInstance(x).setAttributeValue(attribute, instance.doubleValue(attribute)*mult+add);
       			 //System.out.println("instance after="+copy.getInstance(x).doubleValue(attribute));

			 }
			 else if  (attvalue>=thresholds[0]&&attvalue<thresholds[1])
			 {				 

				 mult=Math.pow(2, -2);
         		 add=(1-Math.pow(2, -2))*globalmean+(1*globalsd);
         		 //System.out.println("mult="+mult+" and add="+add);
      			 //System.out.println("instance before="+instance.doubleValue(attribute));
      			 
                 copy.getInstance(x).setAttributeValue(attribute, instance.doubleValue(attribute)*mult+add);
       			 //System.out.println("instance after="+copy.getInstance(x).doubleValue(attribute));

			 }
			 else if  (attvalue>=thresholds[1]&&attvalue<thresholds[2])
			 {				

				 mult=Math.pow(2, -1);
         		 add=(1-Math.pow(2, -1))*globalmean+(1*globalsd);
         		//System.out.println("mult="+mult+" and add="+add);
     			// System.out.println("instance before="+instance.doubleValue(attribute));
                 copy.getInstance(x).setAttributeValue(attribute, instance.doubleValue(attribute)*mult+add);
       			 //System.out.println("instance after="+copy.getInstance(x).doubleValue(attribute));

			 }
			 else
			 {				 

				 mult=Math.pow(2, 2);
         		 add=(1-Math.pow(2, 2))*globalmean+(1*globalsd);
         		//System.out.println("mult="+mult+" and add="+add);
     			 //System.out.println("instance before="+instance.doubleValue(attribute));
                 copy.getInstance(x).setAttributeValue(attribute, instance.doubleValue(attribute)*mult+add);
       			 //System.out.println("instance after="+copy.getInstance(x).doubleValue(attribute));

			 }			 
         }
            return copy;
    }    
    
    /**
     * Units for the parameter of this bias.
     * @return "Standard Deviations"
     */
    public @Override String getUnit(int whichUnit)
    {
    	if (whichUnit==0)
    		return "Standard Deviations";
    	else
    		return "Only one unit in this bias";
    }
    
    public String getLongDescription()
    {
        return "Non Linear Shift";
    }
   
    /**
     * Display-friendly name of this bias.
     * @return "Covariate Shift"
     */
    public String getName() 
    { 
    	return "Non-Linear Shift"; 
    }
    public Dataset injectRandomBias(Dataset original, int attribute) {
		// TODO Auto-generated method stub
		return null;
	}
}
