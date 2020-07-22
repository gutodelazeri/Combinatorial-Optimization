from docplex.mp.model import *
import itertools
import Instance as inst


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

    def __init__(self, instanceName, timeLimit, verbose):
        self._model = None
        self._modelVars_x = None
        self._modelVars_y = None
        self._modelSolution = None
        self._instance = inst.Instance(instanceName)
        self._timeLimit = timeLimit
        self._verbose = verbose

    def _initSolver(self):
        self._model = Model(name=self._instance.name)
        self._model.context.solver.log_output = self._verbose
        self._model.set_time_limit(self._timeLimit)

    def _addVariables(self):
        indices = [(i, j) for i in range(self._instance.n) for j in range(self._instance.m)]
        self._modelVars_x = self._model.binary_var_dict(indices, name="x")
        self._modelVars_y = self._model.continuous_var(lb=0, name="y")

    def _addObjectiveFunction(self):
        self._model.minimize(self._modelVars_y)

    def _addConstraints(self):

        x = self._modelVars_x
        y = self._modelVars_y
        n = self._instance.n
        m = self._instance.m
        p = self._instance.p

        for i in range(n):
            self._model.add_constraint(self._model.sum(x[i, j] for j in range(m)) == 1)

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

    def getSolutionStatus(self):
        return

    # todo treat infeasibility
    def getObjectiveValue(self):
        return self._modelSolution.get_objective_value()

    def getOperatorsPermutation(self):
        return

    # todo treat infeasibility
    def getTasksPartition(self):
        partition = []
        for j in range(self._instance.m):
            partition.append(
                [i for i in range(self._instance.n) if self._modelSolution.get_value(self._modelVars_x[i, j]) == 1])
        return partition
