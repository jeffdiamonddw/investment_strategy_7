from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.duplicate import DefaultDuplicateElimination
from pymoo.core.population import Population
from pymoo.optimize import minimize
from pymoo.util.display.multi import MultiObjectiveOutput
# from pymoo.util.output import MultiObjectiveOutput
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.normalization import normalize
from pymoo.util.roulette import RouletteWheelSelection
from sklearn.cluster import KMeans
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2

from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.operators.selection.tournament import compare
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.operators.sampling.rnd import FloatRandomSampling


from pysamoo.core.algorithm import SurrogateAssistedAlgorithm

import numpy as np


from pymoo.core.population import Population
from pymoo.operators.selection.tournament import TournamentSelection

from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival

import numpy as np
from pymoo.core.population import Population
from pymoo.operators.selection.tournament import TournamentSelection

import copy

from botorch.acquisition.multi_objective import qExpectedHypervolumeImprovement
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
import torch

from pymoo.core.population import Population
from pymoo.core.individual import Individual

import time
import pandas as pd




def swap_individuals(master_pop, recommendations):
    # Get the sizes
    n_master = len(master_pop)
    n_rec = len(recommendations)
    
    if n_rec > n_master:
        raise ValueError("More recommendations than master population size.")
    
    # 1. Choose unique random indices from master_pop to replace
    # replace=False ensures we don't pick the same index twice
    swap_indices = np.random.choice(n_master, n_rec, replace=False)
    
    # 2. Perform the swap
    # Since master_pop is a pymoo Population object, we can 
    # replace the individuals at the chosen indices directly.
    for i, idx in enumerate(swap_indices):
        master_pop[idx] = recommendations[i]
        
    return master_pop


import numpy as np

class SigmoidFunction:
    def __init__(self, min_point, max_point, tolerance):
        self.tol = tolerance
        
        # 1. Define targets explicitly based on the point definitions
        y_min = self.tol          # Target for min_point
        y_max = 1.0 - self.tol    # Target for max_point
            
        # 2. Logit transforms
        logit_min = np.log(y_min / (1.0 - y_min))
        logit_max = np.log(y_max / (1.0 - y_max))
        
        # 3. Calculate slope (k) and midpoint (xm)
        # This naturally handles both increasing (k > 0) and decreasing (k < 0) slopes
        self.k = (logit_max - logit_min) / (max_point - min_point)
        self.xm = max_point - (logit_max / self.k)
      

    def __call__(self, x):
        # Standard Sigmoid formula
        return 1.0 / (1.0 + np.exp(-self.k * (x - self.xm)))



class UCBProblem(Problem):
    def __init__(self, surrogate_problem, beta):
        super().__init__(
            n_var=surrogate_problem.n_var,
            n_obj=surrogate_problem.n_obj,
            xl=surrogate_problem.xl,
            xu=surrogate_problem.xu,
            elementwise_evaluation=False
        )
        self.surrogate_problem = surrogate_problem
        self.beta = beta

    def _evaluate(self, x, out, *args, **kwargs):
        # 1. Get the raw predictions from your SurrogateProblem
        out_surrogate = {}
        self.surrogate_problem._evaluate(x, out_surrogate)
        
        mu = out_surrogate["F"]
        sigma = out_surrogate["S"]
        
        # 2. Apply the UCB acquisition function
        # Higher beta -> Higher reliance on sigma (exploration)
        result = (1- self.beta) * mu - self.beta * sigma
        out["F"] = result

    def avg_uncertainty(self, n_samples=1000):
        # 1. Sample the search space
        X_test = np.random.uniform(self.xl, self.xu, size=(n_samples, self.n_var))
        
        # 2. Get predictions and uncertainty (S) from the surrogate
        # Note: Use your SurrogateProblem's internal eval logic
        out = {}
        self.surrogate_problem._evaluate(X_test, out)
        sigma = out["S"] # Shape (N, n_obj)
        
        # 3. Decision metric:
        # If the average uncertainty (sigma) across the space is below a 
        # threshold, the surrogate has "seen enough."
        avg_uncertainty = np.mean(sigma)
        
        
        return avg_uncertainty
    

def tell_algorithm_with_arrays(algorithm, X, F):
    # 1. Create a list of individuals from the arrays
    # pymoo's Population expects a list of Individual objects
    pop_list = []
    for i in range(len(X)):
        ind = Individual(X=X[i], F=F[i])
        pop_list.append(ind)
    
    # 2. Convert the list into a Pymoo Population object
    bootstrap_pop = Population.new(individuals=pop_list)
    
    # 3. Tell the algorithm
    # This integrates these points into the NSGA2 internal population
    algorithm.tell(bootstrap_pop)
    
    return bootstrap_pop

class APSANGSA2:

    def __init__(self,
                 pop_size=188,
                 n_infills = 180,
                 surr_n_gen=30,
                 surr_thresholds = {'min_point': .1, 'max_point': .01},
                 surr_eps_elim=1e-6,
                 output=MultiObjectiveOutput(),
                 surr_tolerance = .1,
                 surrogate_problem = None,
                 master_problem = None,
                 surrogate_memory_generations = 3,
                 **kwargs):

        
        self.pop_size = pop_size
        self.n_infills = n_infills
        self.surr_n_gen = surr_n_gen
        self.surr_thresholds = surr_thresholds
        self.surr_eps_elim = surr_eps_elim
        self.surr_tolerance = surr_tolerance
        self.sigmoid_function = SigmoidFunction(tolerance = self.surr_tolerance, **self.surr_thresholds)
        self.surrogate_problem = surrogate_problem
        self.master_problem = master_problem
        self.surrogate_memory_generations = surrogate_memory_generations
        
        self.master_algorithm = NSGA2(pop_size = n_infills)
        self.master_algorithm.setup(self.master_problem)
    

        
        self.min_recommendations = np.maximum(0, self.pop_size - self.n_infills)
        self.num_recommendations = self.min_recommendations



        self.archive = Population()

   

    
   
    def _initialize_advance(self, infills=None, **kwargs):
        super()._initialize_advance(infills, **kwargs)
        

   

    def get_guaranteed_candidates(self, pop, n_required):
        # 1. Perform standard duplicate elimination
        dedup = DefaultDuplicateElimination(epsilon=self.surr_eps_elim)
        pop = dedup.do(pop, self.archive)
        
        # 2. If we are under the quota, supplement until we hit the minimum
        if len(pop) < n_required:
            n_missing = n_required - len(pop)
            
            # Generate new random candidates
            sampling = FloatRandomSampling()
            missing_pop = sampling.do(self.surrogate_problem, n_missing)
            
            # Evaluate the new candidates so they are usable
            self.surrogate_problem.evaluate(missing_pop)
            
            # Merge (pop + missing_pop results in a larger list of individuals)
            # We do NOT truncate here, allowing it to be >= n_required
            pop = pop + missing_pop
            
        return pop


    def ask(self):

            
            
            pop = self.master_algorithm.ask()
            
            if self.num_recommendations > 0:
                
                
                if self.surrogate_problem.model.is_bayesian:
                    

                    # 1. Prepare your data for BoTorch
                    train_F = self.archive[-(self.pop_size * self.surrogate_memory_generations):].get("F")

                    # 2. Define the Reference Point (the "worst" acceptable performance)
                    # A simple heuristic is to take the max of each objective + a small buffer
                    if hasattr(self.master_problem, 'ref_point'):
                        ref_point = self.master_problem.ref_point
                    else:
                        ref_point = 1.25 * self.archive.get("F").max(dim=0).values 

                    # 3. Create the Partitioning (required for EHVI)
                    # This identifies the "dominated space" to calculate potential improvement
                    partitioning = NondominatedPartitioning(ref_point=ref_point, Y=train_F)

                    # 4. Instantiate the Acquisition Function
                    acq_func = qExpectedHypervolumeImprovement(
                        model=self.surrogate_problem.model,
                        ref_point=ref_point,
                        partitioning=partitioning,
                    )

                    # 5. Optimize the Acquisition Function directly
                    # This produces 'num_recommendations' worth of points in one go
                    candidate_tensor, _ = optimize_acqf(
                        acq_function=acq_func,
                        bounds=torch.tensor([self.xl, self.xu]).float(), # Your variable bounds
                        q=self.num_recommendations,
                        num_restarts=10,
                        raw_samples=512,
                    )

                    X_infill_np = candidate_tensor.detach().cpu().numpy()
                    candidates = Population.new("X", X_infill_np)


                else:
                    surrogate_optimizer = NSGA2(pop_size=self.n_infills)
            
                    res = minimize(
                        self.surrogate_problem,
                        surrogate_optimizer,
                        ('n_gen', self.surr_n_gen),
                        seed=1,
                        verbose=False,
                        pop=copy.deepcopy(pop) # Only clone the population data
                    )
                    candidates = res.pop
                
                cand = self.get_guaranteed_candidates(candidates, self.min_recommendations)
            
                
            
                if len(cand) <= self.num_recommendations:
                    recommendations = cand
                else:

                    ideal = res.opt.get("F").min(axis=0)
                    nadir = res.opt.get("F").max(axis=0) + 1e-16
                    vals = normalize(cand.get("F"), ideal, nadir)

                    kmeans = KMeans(n_clusters=self.num_recommendations, random_state=0).fit(vals)
                    groups = [[] for _ in range(self.num_recommendations)]
                    for k, i in enumerate(kmeans.labels_):
                        groups[i].append(k)

                    S = []

                    for group in groups:
                        if len(group) > 0:
                            fitness = cand[group].get("crowding").argsort()
                            selection = RouletteWheelSelection(fitness, larger_is_better=False)
                            I = group[selection.next()]
                            S.append(I)

                    recommendations = Population.new(X=cand[S].get("X"))
                
                min_recommendation_pop = recommendations[:self.min_recommendations]
                swapped_pop = swap_individuals(pop, recommendations[self.min_recommendations:])
                infills = Population.merge(min_recommendation_pop, swapped_pop)
            else:
                infills = pop

            
            return infills
    

        
    
    def tell(self, infills, **kwargs):
        
       # Merge existing archive with new infills
        self.archive = Population.merge(self.archive, infills)
            
            
       
        X_infills = infills.get("X")
        if self.master_problem.generation == 0:
            self.surrogate_problem.fit(X_infills, infills.get("F"))    
        
        else:
            F_val = self.surrogate_problem.evaluate(X_infills, return_values_of=["F"])
            F_master = infills.get("F")
            
            # Compute error using the arrays directly
            validation_error = np.mean(np.abs(F_master - F_val) / (np.abs(F_master) + 1e-6))
            self.num_recommendations = self.min_recommendations + int( (self.pop_size - self.min_recommendations) * self.sigmoid_function(validation_error))
            
            self.surrogate_problem.fit(X_infills, F_master)
            F_train = self.surrogate_problem.evaluate(X_infills, return_values_of=["F"])
            train_error = np.mean(np.abs(F_master - F_train) / (np.abs(F_master) + 1e-6))

            print("surrogate error: {}/{}, num_recommendations: {}".format(train_error, validation_error, self.num_recommendations))
       

        

        if hasattr(self.master_problem, "quality"):
            quality = self.master_problem.quality(infills)
        else:
             quality = None     
        
        
        self.master_algorithm.tell(infills)


       




    def _advance(self, infills=None, **kwargs):
        super()._advance(infills, **kwargs)

    def _set_optimum(self):
        nds = NonDominatedSorting().do(self._archive.get("F"), only_non_dominated_front=True)
        self.opt = self._archive[nds]


 
    

    
      
