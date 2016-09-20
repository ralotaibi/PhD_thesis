package meka.classifiers.multilabel.LaCovaC;

 /*
 *    Author Reem Alotaibi to handle multi-label data, 2016
 *	  ra12404@bristol.a.cuk
 */
 
public class LabelList
{
	public int indices[][];  
	public int names[][]; 

	public LabelList()
	{

	} 
	public LabelList(int indices[][], int names[][])
	{
		this.indices = indices;
		this.names = names;
	} 
	
}
