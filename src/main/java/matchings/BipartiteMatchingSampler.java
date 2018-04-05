package matchings;

import java.util.Collections;
import java.util.List;

import bayonet.distributions.Multinomial;
import bayonet.distributions.Random;
import blang.core.LogScaleFactor;
import blang.distributions.Generators;
import blang.mcmc.ConnectedFactor;
import blang.mcmc.SampledVariable;
import blang.mcmc.Sampler;

/**
 * Each time a Permutation is encountered in a Blang model, 
 * this sampler will be instantiated. 
 */
public class BipartiteMatchingSampler implements Sampler {
  /**
   * This field will be populated automatically with the 
   * permutation being sampled. 
   */
  @SampledVariable BipartiteMatching matching;
  /**
   * This will contain all the elements of the prior or likelihood 
   * (collectively, factors), that depend on the permutation being 
   * resampled. 
   */
  @ConnectedFactor List<LogScaleFactor> numericFactors;

  @Override
  public void execute(Random rand) {
	  int n_comp = this.matching.componentSize();
	  int i = rand.nextInt(n_comp);
	  int edge = this.matching.getConnections().get(i);
	  List<Integer> free = this.matching.free2();
	  
	  double logBefore = logDensity();
	  
	  if (edge == BipartiteMatching.FREE) {
		  int j = rand.nextInt(free.size());
		  this.matching.getConnections().set(i, free.get(j));
	  }
	  else {
		  int j = rand.nextInt(free.size() + 1);
		  if(j == free.size()) {
			  this.matching.getConnections().set(i, BipartiteMatching.FREE);
		  }
		  else {
			  this.matching.getConnections().set(i, free.get(j));
		  }
	  }
	  
	  double logAfter = logDensity();  
	  boolean accept = Generators.bernoulli(rand, Math.min(1.0, Math.exp(logAfter - logBefore)));
	  // If not accept, swap back	    	
	  if (!accept) {
		  this.matching.getConnections().set(i, edge);
	}

  }
  
  private double logDensity() {
    double sum = 0.0;
    for (LogScaleFactor f : numericFactors)
      sum += f.logDensity();
    return sum;
  }
}
