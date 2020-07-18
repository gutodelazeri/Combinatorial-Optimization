import glpk
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
        self._model = glpk.LPX()
        self._model.name = self._instance.name
        glpk.env.term_on = self._verbose

    def _addVariables(self):


    def _addObjectiveFunction(self):


    def _addConstraints(self):
        def s(i, k):
            if (i + 1 == k) or (i - 1 == k):
                return 2
            else:
                return 1

        x = self._modelVars_x
        y = self._modelVars_y
        n = self._instance.n
        m = self._instance.m
        p = self._instance.p

        for i in range(n):
            self._model.add_constraint(self._model.sum(x[i, j] for j in range(m)) == 1)

        for i, k in itertools.combinations(range(n), 2):
            for j in range(m):
                self._model.add_constraint(x[i, j] + x[k, j] <= s(i, k))

        for j in range(m):
            self._model.add_constraint(y >= self._model.sum(x[i, j] * p[i][j] for i in range(n)))

    def _buildModel(self):
        self._initSolver()
        self._addVariables()
        self._addObjectiveFunction()
        self._addConstraints()

    def solveModel(self):
        self._model.erase()
        self._buildModel()
        self._modelSolution = self._model.solve()

    def getSolutionStatus(self):
        return

    def getObjectiveValue(self):
        return

    def getOperatorsPermutation(self):
        return

    def getTasksPartition(self):
        return
