package tud.ke.ml.project.classifier;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;


import tud.ke.ml.project.util.Pair;

/**
 * This implementation assumes the class attribute is always available (but probably not set).
 * 
 */
public class NearestNeighbor extends INearestNeighbor implements Serializable {
	private static final long serialVersionUID = 1L;


	protected double[] scaling;
	protected double[] translation;


	protected List<List<Object>> trainingdata;//new creat

	@Override
	public String getMatrikelNumbers() {
		return "2673707,2374268,2567884";
	}

	@Override
	protected void learnModel(List<List<Object>> data) {
		  trainingdata = data;
	}

	@Override
	protected Map<Object, Double> getUnweightedVotes(List<Pair<List<Object>, Double>> subset) {
		Map<Object, Double> unweightedVotes = new HashMap<Object, Double>();
		int classAttribute = getClassAttribute();
		for(int i=0; i<subset.size();i++){
			if(unweightedVotes.containsKey(subset.get(i).getA().get(classAttribute))){
				unweightedVotes.replace(subset.get(i).getA().get(classAttribute),unweightedVotes.get(subset.get(i).getA().get(classAttribute))+1.0);
			}else{
				unweightedVotes.put(subset.get(i).getA().get(classAttribute), 1.0);
			}
		}
		return unweightedVotes;
	}

	@Override
	protected Map<Object, Double> getWeightedVotes(List<Pair<List<Object>, Double>> subset) {
		Map<Object, Double> weightedVotes = new HashMap<Object, Double>();
		int classAttribute = getClassAttribute();
		for(int i=0; i<subset.size();i++){
			if(weightedVotes.containsKey(subset.get(i).getA().get(classAttribute))){
				weightedVotes.replace(subset.get(i).getA().get(classAttribute),weightedVotes.get(subset.get(i).getA().get(classAttribute))+1.0/subset.get(i).getB());
			}else{
				weightedVotes.put(subset.get(i).getA().get(classAttribute), 1.0/subset.get(i).getB());
			}
		}
		return weightedVotes;
	}

	@Override
	protected Object getWinner(Map<Object, Double> votes) {
		Iterator<Map.Entry<Object, Double>> it = votes.entrySet().iterator();
		Object winner = null;
		double max=0;
		int classAttribute = getClassAttribute();
		ArrayList<Object> winnerlist = new ArrayList<Object>();
		while(it.hasNext()){
			Map.Entry<Object, Double> pair = (Map.Entry<Object, Double>)it.next();
			if(pair.getValue() > max){
				winnerlist.clear();
				winnerlist.add(pair.getKey()); 
				max = pair.getValue();
				
			}else{
				if(pair.getValue() == max){
					winnerlist.add(pair.getKey());
				}
			}
		}
		if(winnerlist.size()==1){
			return winnerlist.get(0);
		}
		else{ //if have more than one class
			int[] a = new int[winnerlist.size()];
			for(int i=0;i<winnerlist.size();i++){
				for(int j=0;j<trainingdata.size();j++){
					if(winnerlist.get(i)==trainingdata.get(j).get(classAttribute)){
						a[i] = j;
						break;
					}
				}		
			}
			int minvalue = a[0];  // return the first appeared class in trainingdata-list
		    for (int x = 1; x < a.length; x++) {  
		        if (a[x] > minvalue)  
		        	minvalue = a[x];  
		    }  
			for(int i=0;i<trainingdata.size();i++){  
	            if(a[i]==minvalue){  
	                winner = winnerlist.get(i);  
	            }
			}
			return winner;
		}
	}

	@Override
	protected Object vote(List<Pair<List<Object>, Double>> subset) {
		Object winner = new Object();
		if(isInverseWeighting()){//WeightedVotes
			Map<Object, Double> weightedVotes = getWeightedVotes(subset);
			winner = getWinner(weightedVotes);
		}
		else{//UnweightedVotes
			Map<Object, Double> unweightedVotes = getUnweightedVotes(subset);
			winner = getWinner(unweightedVotes);	
		}
		return winner;
	}

	@Override
	protected List<Pair<List<Object>, Double>> getNearest(List<Object> data) {
		if(isNormalizing()){
		       double[][] normalization = normalizationScaling();
			   scaling=normalization[0];
			   translation=normalization[1];
			}
		List<Pair<List<Object>, Double>> nearest = new ArrayList<Pair<List<Object>, Double>>(trainingdata.size());
		if(getMetric()==1){
			for(int i=0;i<trainingdata.size();i++){
				nearest.add(new Pair<List<Object>, Double>(trainingdata.get(i), determineEuclideanDistance(data,trainingdata.get(i))));
			}
		}
		else{
			for(int i=0;i<trainingdata.size();i++){
				nearest.add(new Pair<List<Object>, Double>(trainingdata.get(i), determineManhattanDistance(data,trainingdata.get(i))));
		    }
		}
		Collections.sort(nearest,new Comparator<Pair<List<Object>, Double>>(){//sort the nearest-list
            public int compare(Pair<List<Object>, Double> pair0, Pair<List<Object>, Double> pair1) {
            	if(pair0.getB()>pair1.getB()){ 
					return 1;
				}
				else if(pair0.getB()==pair1.getB()) {
					return 0;
				}
				else {
					return -1;  
				}
             }
            });
		int k = getkNearest();
		for(int i = nearest.size() - 1; i >= k; i--){
			nearest.remove(i);// if more than one instances have a same distance, we just remove the instances after the k-th of nearest-list.
		}
		return nearest;
	}

	@Override
	protected double determineManhattanDistance(List<Object> instance1, List<Object> instance2) {
		double manhattanDistance = 0;
		int classAttribute = getClassAttribute();
		
		for(int i = 0; i < instance1.size(); i++){
			if(i != classAttribute){
				if(instance1.get(i) instanceof String){
				    if(!(instance1.get(i).equals(instance2.get(i)))){
					    manhattanDistance += 1.0;
				    }
			    }
			    else{
				    if(isNormalizing()){
				    	manhattanDistance += Math.abs(((double)instance1.get(i)-translation[i]) - ((double)instance2.get(i)-translation[i]))/scaling[i];
			        }
				    else{
				    	manhattanDistance += Math.abs((double)instance1.get(i) - (double)instance2.get(i));
				    }
			    }
			}	
		}	
		return manhattanDistance;
	}

	@Override
	protected double determineEuclideanDistance(List<Object> instance1, List<Object> instance2) {
		double EuclideanDistance = 0;	
		int classAttribute = getClassAttribute();
		
		for(int i = 0; i < instance1.size(); i++){
			if(i!=classAttribute){
				if(instance1.get(i) instanceof String){
					if(!(instance1.get(i).equals(instance2.get(i)))){
						EuclideanDistance += 1.0;
					}
				}
				else{
					if(isNormalizing()){
						EuclideanDistance += Math.pow(Math.abs(((double)instance1.get(i)-translation[i]) - ((double)instance2.get(i)-translation[i]))/scaling[i],2);
					}
					else{
						EuclideanDistance += Math.pow((double)instance1.get(i) - (double)instance2.get(i),2);
					}
					EuclideanDistance = Math.sqrt(EuclideanDistance); 
				}
			}
		}
		return EuclideanDistance;    
	}

	@Override
	protected double[][] normalizationScaling() {
		double[][] normalization = new double[2][trainingdata.get(0).size()];
		double max=Double.MIN_VALUE;
		double min=Double.MAX_VALUE;
		for(int i=0;i<trainingdata.get(0).size();i++){
			if(trainingdata.get(0).get(i) instanceof Double){
				for(int x=0;x<trainingdata.size();x++){
					if((double)trainingdata.get(x).get(i)> max){
						max = (double)trainingdata.get(x).get(i);
					}
					if((double)trainingdata.get(x).get(i)< min){
						min = (double)trainingdata.get(x).get(i);
					}
				}
				normalization[0][i]=max-min;
				normalization[1][i]=min;
				max=Double.MIN_VALUE;
				min=Double.MAX_VALUE;		
			}
			else{
				normalization[0][i]=1.0;
				normalization[1][i]=1.0;
			}
		}
		return normalization;
	}

}
