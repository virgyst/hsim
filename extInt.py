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

class Problem(ElementwiseProblem):
    def __init__(self):
        self.lab = lav.Lab()
        super().__init__(n_var=10, n_obj=1)
    
    def _evaluate(self, x, out, *args, **kwargs):
        lista = lav.batchCreate(1, numJobs=10)
        sorted_lista = [lista[i] for i in x]
        self.lab.gate.Store.items = copy.copy(sorted_lista)
        self.lab.run(1100)
        df = self.lab.env.state_log
        Cmax = self.lab.calculate_makespan(df)
        print(f"makespan: {Cmax}")
        out["F"] = Cmax

problem = Problem()

algorithm = GA(
    pop_size=20,
    sampling=PermutationRandomSampling(),
    mutation=InversionMutation(),
    crossover=OrderCrossover(),
    eliminate_duplicates=True
)

res = minimize(problem, algorithm, seed=1, verbose=False)

# print(f"Best solution found: {res.X}")
# print(f"Function value: {res.F}")


