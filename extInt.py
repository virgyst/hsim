import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.problems.single.traveling_salesman import create_random_tsp_problem
from pymoo.operators.sampling.rnd import PermutationRandomSampling
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.termination.default import DefaultSingleObjectiveTermination

# from ... import Lab, classi varie, etc

class Problem(ElementwiseProblem):
    def __init__(self):
        # configuro il problema
        # creo lab
        # n_var = # jobs

        super().__init__(n_var=10, n_obj=1)
    def _evaluate(self, x, out, *args, **kwargs):
        lista=lab.gate.store.items()
        sorted_lista=[LISTA[i] for i in x] 
        # passo al Lab la var "x" che è la sequenza da utilizzare
        # sorto secondo x dove x = np.array([1,2,3,4,...])
        # eseguo codice per simulazione --> makespan
        # ci metto il modello che funziona SENZA RL
        Cmax = 10
        out["F"] = Cmax

problem = Problem()

algorithm = GA(
    pop_size=20,
    sampling=PermutationRandomSampling(),
    mutation=InversionMutation(),
    crossover=OrderCrossover(),
    eliminate_duplicates=True
)

res = minimize(problem,
               algorithm,
               seed=1,
               verbose=False)


