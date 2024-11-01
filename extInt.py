import numpy as np
import hsim.core.pymulate as pym
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import PermutationRandomSampling
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.termination.default import DefaultSingleObjectiveTermination
import time
import copy
import lavoro as lav
import pandas as pd

class Problem(ElementwiseProblem):
    def __init__(self):
        
        super().__init__(n_var=10, n_obj=1)
    
    def _evaluate(self, x, out, *args, **kwargs):
        self.lab = lav.Lab()
        lista = lav.batchCreate(1, numJobs=10)
        sorted_lista = [lista[i] for i in x]
        self.lab.gate.Store.items = copy.copy(sorted_lista)
        self.lab.run(1100)
        df = self.lab.env.state_log
        # self.env.state_log
        df=pd.DataFrame(self.lab.env.state_log,columns=['Resource','ResourceName','State','StateName','entity','store','timeIn','timeOut'])
        Cmax = self.lab.calculate_makespan(df)
        #Cmax = df.timeOut.max()
        # print(f"makespan: {Cmax}")
        out["F"] = Cmax

problem = Problem()

algorithm = GA(
    pop_size=20,
    sampling=PermutationRandomSampling(),
    mutation=InversionMutation(),
    crossover=OrderCrossover(),
    eliminate_duplicates=True
)

res = minimize(problem, algorithm, seed=1, verbose=True)

# print(f"Best solution found: {res.X}")
print(f"Function value: {res.F}")


