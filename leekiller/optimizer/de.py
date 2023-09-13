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
        self.log_objective = [] 
        self.log_control_params = []
        self.log_info = []
        self.log_op_objective = 0
        self.log_op_control_param = {}
        self.log_op_info = {}

    def run(self, itr: int=60, batch: int=8):
        # itr: iteration times
        check_frequecy = itr/10
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
                    update, update_vector, update_objective, update_info = zip(*pool.map(self._iteration, sampled_index))
                self.log_objective.append(update_objective)
                self.log_control_params.append(update_vector)
                self.log_info.append(update_info)
                for j in range(len(update)):
                    if update[j]:
                        if update_objective[j] > self.log_op_objective:
                            self.log_op_objective = update_objective[j]
                            self.log_op_control_param = update_vector[j]
                            self.log_op_info = update_info[j]
                            np.savez("out-op_control_param.npz", **self.log_op_control_param)
                            print("# %s/%s iteration, optimized objective: %.6f" %(i+j, itr, self.log_op_objective))
                            print("op_control_param:", self.log_op_control_param)
                            self._print_info(self.log_op_info)
                            print("\n")
                np.savez("out.npz", 
                         populations=self.populations,
                         objective=self.log_objective, 
                         control_params=self.log_control_params,
                         info = self.log_info,
                         op_objective=self.log_op_objective,
                         op_info = self.log_op_info)
                i = i + batch
                if i % check_frequecy == 0 and not any(update):
                    print("# %s/%s iteration, optimized objective: %.6f" %(i, itr, self.log_op_objective))
                    print("op_control_param:", self.log_op_control_param, "\n")
            print('Finish %s iterations, optimized objective: %.6f' %(i, self.log_op_objective))
            print('op_control_param:', self.log_op_control_param)
            self._print_info(self.log_op_info)
            print('==============================')            

    def load_data(self, data_path: str=None):
        """ Load market data to self.dataframe

        """
        raise NotImplementedError("No market data!")

    def get_objective_value(self, control_params: dict) -> tuple[float, dict]:
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

    def _print_info(self, info: dict):
        if info:
            total_sesssions = info["Summary"]["total_sessions"]
            print(info["Summary"])
            for i in range(total_sesssions):
                print("num_trade: {}, roi: {:.2f}%,  winrate: {:.2f}%, drawdown: {:.2f}%, sharp: {:.2f}."
                      .format(info["num_trade"][i],
                              info["roi"][i] * 100,
                              info["winrate"][i] * 100,
                              info["drawdown"][i],
                              info["sharp"][i]))
        else:
            print("No update.")
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
        update, update_vector, update_objective, update_info = self._selection(di_index, ci_vector)
        if update:
            self.populations[di_index] = ci_vector
        return update, update_vector, update_objective, update_info

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
            ci_vector[key] = np.asarray(ci_vector[key])
        return ci_vector

    def _selection(self, di_index, ci_vector: dict):
        """ DE selection

        """
        di_vector = self.populations[di_index]
        di_objective, di_info = self.get_objective_value(di_vector)
        ci_objective, ci_info = self.get_objective_value(ci_vector)
        update = False
        update_vector = None
        update_objective = None
        update_info = None
        if di_objective < ci_objective:
            update = True
            update_vector = ci_vector
            update_objective = ci_objective
            update_info = ci_info
        else:
            update = False
        return update, update_vector, update_objective, update_info
