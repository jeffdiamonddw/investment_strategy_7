import random
import time
import numpy as np
from utils import get_dna_hash


class PerturbationExecutor:
    def __init__(self, workhorse_cls, workhorse_args, n_samples=3, perturbation_cv=0.01):
        """
        An 'Abstract-Aware' wrapper. It stores the CLASS of the simulation 
        to be run, not the object.
        """
        # 1. Instantiate the simulation locally to determine dimensions
        self.local_sim = workhorse_cls(**workhorse_args)
        self.n_samples = n_samples
        self.perturbation_cv = perturbation_cv
        self.workhorse_args = workhorse_args
      
        

    def evaluate(self, x):
        # Implementation logic for running 5 simulations and averaging[cite: 1]
        sim_id = get_dna_hash(x)
        results = []
        for i in range(self.n_samples):
            # Apply Gaussian perturbation based on input CV[cite: 1]
            noise = np.random.normal(0, self.perturbation_cv * np.abs(x), size=x.shape)
            perturbed_x = x + noise
            
            # Route evaluation to the locally held simulation object
            sim_id = get_dna_hash(x)
            result = self.local_sim.evaluate(perturbed_x)
            print("done perturbation {} for {} at {}".format(i, sim_id, time.ctime()), flush = True)
            results.append(result)

        
        time.sleep(random.uniform(0, 5))
        return results