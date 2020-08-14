class Individual:
    def __init__(self, permutation, intervals, fitness=None):
        self.permutation = permutation
        self.intervals = intervals
        self.fitness = fitness

    def __eq(self, e):
        if self.permutation == e.permutation and self.intervals == e.intervals:
            return True
        else:
            return False
