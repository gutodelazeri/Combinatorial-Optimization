import sys
import time
from docplex.mp.model import *
from itertools import combinations
from Instance import Instance
from Statistics import Statistics


class IPSolver:
    """
    An IP solution for an instance of the TB problem

    ...

    Arguments
    ----------
    instanceName: str
        path for the file of an instance

    timeLimit: int
        time limit for the solver (in seconds)

    verbose: bool
        enable/disable solver verbosity

    Attributes
    ----------
    _model : CPLEX model object
    _modelVars_x : CPLEX variable object
    _modelVars_y : CPLEX variable object
    _modelSolution : CPLEX solution object
    _instance : Instance object
    _timeLimit : int
    _verbose : bool
    """

    def __init__(self, instanceName, timeLimit=1800, verbose=True):
        self._model = None
        self._modelVars_x = None
        self._modelVars_w = None
        self._modelVars_y = None
        self._modelSolution = None
        self._instance = Instance(instanceName)
        self._timeLimit = timeLimit
        self._verbose = verbose
        self._stats = Statistics(instanceName, "Integer Programming")

    def _initSolver(self):
        self._model = Model(name=self._instance.name)
        self._model.context.solver.log_output = self._verbose
        self._model.set_time_limit(self._timeLimit)

    def _addVariables(self):
        x_indices = [(i, j) for i in range(self._instance.n) for j in range(self._instance.m)]
        w_indices = [(i, j, k) for i, j in combinations(range(self._instance.n), 2) for k in range(self._instance.m)]

        self._modelVars_x = self._model.binary_var_dict(x_indices, name="x")
        self._modelVars_w = self._model.binary_var_dict(w_indices, name="w")
        self._modelVars_y = self._model.continuous_var(lb=0, name="y")

    def _addObjectiveFunction(self):
        self._model.minimize(self._modelVars_y)

    def _addConstraints(self):

        x = self._modelVars_x
        w = self._modelVars_w
        y = self._modelVars_y
        n = self._instance.n
        m = self._instance.m
        p = self._instance.p
        model = self._model

        for i in range(n):
            self._model.add_constraint(model.sum(x[i, j] for j in range(m)) == 1)

        # for j in range(m):
        #    self._model.add_constraint(model.sum(x[i, j] for i in range(n)) >= 1)

        for i, j in combinations(range(n), 2):
            for k in range(m):
                model.add_constraint(w[i, j, k] <= (x[i, k] + x[j, k]) / 2)
                model.add_constraint(w[i, j, k] >= x[i, k] + x[j, k] - 1)
                model.add_constraint(w[i, j, k] <= x[j - 1, k])

        for j in range(m):
            self._model.add_constraint(y >= self._model.sum(x[i, j] * p[i][j] for i in range(n)))

    def _buildModel(self):
        self._initSolver()
        self._addVariables()
        self._addObjectiveFunction()
        self._addConstraints()

    def solveModel(self):
        if self._model is not None:
            self._model.clear()
        self._buildModel()
        self._modelSolution = self._model.solve()

    def getStatistics(self):
        return self._stats

    def getSolutionStatus(self):
        return self._modelSolution is not None

    def getObjectiveValue(self):
        if self.getSolutionStatus():
            return self._modelSolution.get_objective_value()
        else:
            return -1

    def getRelativeGap(self):
        if self.getSolutionStatus():
            return self._model.solve_details.mip_relative_gap
        else:
            return -1

    def getTasksPartition(self):
        if self.getSolutionStatus():
            intervals = []
            x = self._modelVars_x
            sol = self._modelSolution
            n = self._instance.n
            for operator in self.getOperatorsPermutation():
                intervals.append([task for task in range(n) if sol.get_value(x[task, operator]) == 1])
            return intervals
        else:
            return []

    def getOperatorsPermutation(self):
        if self.getSolutionStatus():
            permutation = []
            for task in range(self._instance.n):
                for operator in range(self._instance.m):
                    if operator not in permutation and self._modelSolution.get_value(
                            self._modelVars_x[task, operator]) == 1:
                        permutation.append(operator)
            return permutation
        else:
            return []

    def getSolution(self):
        if self.getSolutionStatus():
            n = self._instance.n
            m = self._instance.m
            sol = self._modelSolution
            x = self._modelVars_x
            return [j for i in range(n) for j in range(m) if sol.get_value(x[i, j]) == 1]
        else:
            return []


def test():
    inst = str(sys.argv[1])

    start = time.time()
    ip = IPSolver(inst, 1800, False)
    ip.solveModel()
    end = time.time()

    total = end - start

    with open("../Tests/ip.txt", "a") as output:
        output.write("{0},{1:.2f},{2:.2f}\n".format(inst, total, ip.getObjectiveValue()))


def debug():


if __name__ == "__main__":
    test()
