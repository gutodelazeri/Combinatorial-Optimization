class Individual:
    def __init__(self, permutation, interval, fitness=-1):
        self.permutation = permutation
        self.interval = interval
        self.chromosome = []
        self.fitness = fitness

    def generateChromosome(self):
        chromosome = []
        for operator in self.permutation:
            chromosome.extend([i for i in self.interval if i == operator])
        self.chromosome = chromosome
