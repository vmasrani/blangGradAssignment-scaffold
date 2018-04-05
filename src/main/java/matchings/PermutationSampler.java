package matchings;

import java.util.Collections;
import java.util.List;

import bayonet.distributions.Random;
import blang.core.LogScaleFactor;
import blang.distributions.Generators;
import blang.mcmc.ConnectedFactor;
import blang.mcmc.SampledVariable;
import blang.mcmc.Sampler;
import briefj.collections.UnorderedPair;

/**
 * Each time a Permutation is encountered in a Blang model, 
 * this sampler will be instantiated. 
 */
public class PermutationSampler implements Sampler {
  /**
   * This field will be populated automatically with the 
   * permutation being sampled. 
   */
  @SampledVariable Permutation permutation;
  /**
   * This will contain all the elements of the prior or likelihood 
   * (collectively, factors), that depend on the permutation being 
   * resampled. 
   */
  @ConnectedFactor List<LogScaleFactor> numericFactors;

  @Override
  public void execute(Random rand) {
	int n_comp = this.permutation.componentSize();
    int i = rand.nextInt(n_comp);
    int j = rand.nextInt(n_comp);
    
    double logBefore = logDensity();
    Collections.swap(this.permutation.getConnections(), i, j);
    double logAfter = logDensity();
    boolean accept = Generators.bernoulli(rand, Math.min(1.0, Math.exp(logAfter - logBefore)));
    // If not accept, swap back	    	
    if (!accept) {
    		Collections.swap(this.permutation.getConnections(), i, j);
	}
  }
  
  private double logDensity() {
    double sum = 0.0;
    for (LogScaleFactor f : numericFactors)
      sum += f.logDensity();
    return sum;
  }
}
