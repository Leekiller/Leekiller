import os
import glob
import time
import numpy as np
from multiprocessing import Pool

class DE:
    def __init__(self, objective, control_params, control_params_range, mu=0.5, xi=0.9, c=15):
        # DE parameters
        self.objective = objective
        self.mu = mu    # Mutation factor
        self.xi = xi    # Crossover rate
        self.c = c
        self.k = 0  # Number of control parameters
        for key in control_params:
            self.k += len(control_params[key])  
        self.number_of_populations = self.c * self.k
        self.populations = []
        # Control parameters
        self.control_params = control_params
        self.control_params_range = control_params_range
        # Log data 
        self.log_objective = [[]] * self.number_of_populations
        self.log_control_params = [[]] * self.number_of_populations
        self.log_op_objective = 0
        self.log_op_control_param = []

    def create_populations(self):
        for i in range(self.number_of_populations):
            population = {}
            for key in self.control_params:
                n_control_parameters = len(self.control_params[key])
                population[key] = np.array(np.random.uniform(self.control_params_range[key][0], 
                                                             self.control_params_range[key][1],
                                                             n_control_parameters)).astype(int)
            self.populations.append(population)

    def run(self, itr=60):
        # itr: iteration times
        if len(self.populations) == 0:
            """ To do

            Using raise Error
            """
            print("Please create populations, DE.create_populations(control_params, control_params_range).")
        else:
            i = 0
            print("Start differential evolution...")
            print("Number of control parameters: %s" %self.k)
            print("Populations size: %s" %len(self.populations))
            print("==============================")
            while i < itr:
                with Pool() as pool:
                    """
                    update, update_vector, update_objective = zip(*pool.map(self._iteration,
                                                                            range(self.number_of_populations)))
                    """
                    # update, update_vector, update_objective = zip(*pool.map(self._iteration, range(1)))
                    test_i = self._sample(1)
                    update, update_vector, update_objective = self._iteration(test_i[0])
                    update = [update]
                    update_vector = [update_vector]
                    update_objective = [update_objective]
                for j in range(len(update)):
                    self.log_objective[j].append(update_objective[j])
                    self.log_control_params[j].append(update_vector[j])
                    if update[j]:
                        if update_objective[j] > self.log_op_objective:
                            self.log_op_objective = update_objective[j]
                            self.log_op_control_param = update_vector[j]
                            np.savez("out-op_control_param.npz", **self.log_op_control_param)
                            print("# %s/%s iteration, optimized ROI: %.6f%%" %(i+j, itr, self.log_op_objective))
                np.savez("out.npz", 
                         populations=self.populations,
                         objective=self.log_objective, 
                         control_params=self.log_control_params,
                         op_objective=self.log_op_objective)
                #i = i + self.number_of_populations
                i = i + 1
            print('Finish %s iterations, optimized ROI: %.6f%%' %(i, self.log_op_objective))
            print('==============================')            

    def _objective(self):
        """ Objective function

        Using ROI
        """
        return None

    def _sample(self, n, not_include=[]):
        # Randomly sample n populations for one differential evolution iteration
        choosed = []
        np.random.seed()
        for i in range(n):
            s = int(self.number_of_populations * np.random.uniform(0,1))
            while (s in choosed) or (s in not_include):
                s = int(self.number_of_populations * np.random.uniform(0,1))
            choosed.append(s)
        return np.array(choosed)
    
    def _iteration(self, di_index: int):
        """ DE iteration

        di_index: Index of the origianl vector.
        """
        mi_vector = self._mutation(di_index)
        ci_vector = self._crossover(di_index, mi_vector)
        update, update_vector, update_objective = self._selection(di_index, ci_vector)
        if update:
            self.populations[di_index] = ci_vector
        return update, update_vector, update_objective

    def _mutation(self, di_index: int):
        """ DE mutation

        """
        mi_vector = {}  # Trail vector
        valid_mutation = False
        while not valid_mutation:
            dix_index = self._sample(n=3, not_include=[di_index])
            di1_vector = self.populations[dix_index[0]]
            di2_vector = self.populations[dix_index[1]]
            di3_vector = self.populations[dix_index[2]]
            for key in di1_vector:
                mi_vector[key] = (np.array(di1_vector[key]) +
                                  self.mu*(np.array(di2_vector[key])-np.array(di3_vector[key]))
                                  ).astype(int)
                # Check the validation of mi_vector
                invalid = [temp for temp in mi_vector[key] 
                           if temp < self.control_params_range[key][0] 
                           or temp > self.control_params_range[key][1]
                           ]
                if len(invalid) != 0:
                    valid_mutation = False
                    break
                else:
                    valid_mutation = True
        return mi_vector
    
    def _crossover(self, di_index: int, mi_vector: dict):
        """ DE crossover

        """
        di_vector = self.populations[di_index]
        ci_vector = {}  # Target vector
        for key in di_vector:
            ci_vector[key] = []
            for i in range(len(di_vector[key])):
                r = np.random.uniform(0, 1)
                if r < self.xi:
                    ci_vector[key].append(mi_vector[key][i])
                else:
                    ci_vector[key].append(di_vector[key][i])
        return ci_vector

    def _selection(self, di_index, ci_vector: dict):
        """ DE selection

        """
        di_vector = self.populations[di_index]
        di_objective = self.objective(di_vector)
        ci_objective = self.objective(ci_vector)
        update = False
        update_vector = None
        update_objective = None
        if di_objective < ci_objective:
            update = True
            update_vector = ci_vector
            update_objective = ci_objective
        else:
            update = False
        return update, update_vector, update_objective
