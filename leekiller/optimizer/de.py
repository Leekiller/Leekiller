from multiprocessing import Pool
import numpy as np

class DE:
    def __init__(
            self, 
            control_params: dict=None, 
            control_params_range: dict=None, 
            mu: float=0.5, 
            xi: float=0.9, 
            c: float=15
    ):
        # DE parameters
        self.mu = mu    # Mutation factor
        self.xi = xi    # Crossover rate
        self.c = c
        # Control parameters
        if control_params is not None:
            self.control_params = control_params
        if control_params_range is not None:
            self.control_params_range = control_params_range
        # Generate populations
        self.k = 0  # Number of control parameters
        for key in self.control_params_range:
            self.k += len(self.control_params_range[key])  
        self.number_of_populations = self.c * self.k
        self.populations = []
        self._create_populations()
        # Log
        self.log_objective = [[]] * self.number_of_populations
        self.log_control_params = [[]] * self.number_of_populations
        self.log_op_objective = 0
        self.log_op_control_param = []

    def run(self, itr: int=60, batch: int=8):
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
            print("itr: %s, batch: %s" %(itr, batch))
            print("==============================")
            while i < itr:
                sampled_index = self._sample(batch)
                with Pool() as pool:
                    update, update_vector, update_objective = zip(*pool.map(self._iteration, sampled_index))
                for j in range(len(update)):
                    self.log_objective[j].append(update_objective[j])
                    self.log_control_params[j].append(update_vector[j])
                    if update[j]:
                        if update_objective[j] > self.log_op_objective:
                            self.log_op_objective = update_objective[j]
                            self.log_op_control_param = update_vector[j]
                            np.savez("out-op_control_param.npz", **self.log_op_control_param)
                            print("# %s/%s iteration, optimized session_win_rate: %.6f%%" %(i+j, itr, self.log_op_objective))
                np.savez("out.npz", 
                         populations=self.populations,
                         objective=self.log_objective, 
                         control_params=self.log_control_params,
                         op_objective=self.log_op_objective)
                i = i + batch
            print('Finish %s iterations, optimized session_win_rate: %.6f%%' %(i, self.log_op_objective))
            print('==============================')            

    def load_data(self, data_path: str=None):
        """ Load market data to self.dataframe

        """
        raise NotImplementedError("No market data!")

    def get_objective_value(self, control_params: dict) -> float:
        raise NotImplementedError("Objective value not defined!")

    def _create_populations(self):
        for i in range(self.number_of_populations):
            population = {}
            for key in self.control_params:
                if key in self.control_params_range:
                    n_control_parameters = len(self.control_params[key])
                    population[key] = np.array(np.random.uniform(self.control_params_range[key][0], 
                                                                 self.control_params_range[key][1],
                                                                 n_control_parameters)).astype(int)
                elif (key in self.control_params) and (key not in self.control_params_range):
                    population[key] = self.control_params[key]
            self.populations.append(population)

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
            for key in self.control_params:
                if key in self.control_params_range:
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
                else:
                    mi_vector[key] = di1_vector[key]
        return mi_vector
    
    def _crossover(self, di_index: int, mi_vector: dict):
        """ DE crossover

        """
        di_vector = self.populations[di_index]
        ci_vector = {}  # Target vector
        for key in self.control_params:
            if key in self.control_params_range:
                ci_vector[key] = []
                for i in range(len(di_vector[key])):
                    r = np.random.uniform(0, 1)
                    if r < self.xi:
                        ci_vector[key].append(mi_vector[key][i])
                    else:
                        ci_vector[key].append(di_vector[key][i])
            else:
                ci_vector[key] = di_vector[key]
        return ci_vector

    def _selection(self, di_index, ci_vector: dict):
        """ DE selection

        """
        di_vector = self.populations[di_index]
        di_objective = self.get_objective_value(di_vector)
        ci_objective = self.get_objective_value(ci_vector)
        print("di_obj: %.6f, ci_obj: %.6f" %(di_objective, ci_objective))
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
