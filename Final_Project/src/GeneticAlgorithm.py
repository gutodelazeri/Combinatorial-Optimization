import random as rd
import time
import sys
import math
from Instance import Instance
from Individual import Individual
from Statistics import Statistics


class GeneticAlgorithm:
    def __init__(self, instanceName, Mu, Lambda, k, Phi, Omega, timeLimit=1800, verbose=True):
        self._instance = Instance(instanceName)
        self._mu = Mu
        self._lambda = Lambda
        self._k = k
        self._phi = Phi
        self._omega = Omega
        self._timeLimit = timeLimit
        self._verbose = verbose
        self._stats = Statistics(instanceName, "Genetic Algorithm")
        self._bestIndividual = None

    def _generateInitialPopulation(self):
        def createIntervals(): #todo: deal with the cases when |m-n| is small and when m > n
            delimiters = []
            intervals = []
            while len(delimiters) < self._instance.m:
                n = rd.randint(0, self._instance.n-2)
                while n in delimiters:
                    n = rd.randint(0, self._instance.n-2)
                delimiters.append(n)
            delimiters.sort()
            start = 0
            for delimiter in delimiters:
                intervals.append([i for i in range(start, delimiter+1)])
                start = delimiter+1
            intervals.append([i for i in range(start, self._instance.n)])
            return intervals
        population = []
        permutation = [i for i in range(self._instance.m)]
        for k in range(self._mu):
            intervals = createIntervals()
            rd.shuffle(permutation)
            population.append(Individual(permutation.copy(), intervals))
        return population

    def _randomTournament(self, population):
        if len(population) / 2 >= self._k:
            sample = []
            for _ in range(self._k):
                index = rd.randint(0, self._mu - 1)
                while population[index] in sample:
                    index = rd.randint(0, self._mu - 1)
                sample.append(population[index])
            return self._getFittestIndividual(sample)
        else:
            sample = population.copy()
            sampleSize = len(sample)
            deleted = 0
            while deleted < self._k:
                index = rd.randint(0, sampleSize - 1)
                sample.pop(index)
                sampleSize -= 1
                deleted += 1
            return self._getFittestIndividual(sample)

    def _selectNewPopulation(self, population):
        population.sort(key=lambda individual: individual.fitness)
        population = population[:-self._lambda]
        return population

    def _getFitness(self, individual):
        maxCost = math.inf
        for interval, operator in zip(individual.intervals, individual.permutation):
            cost = 0
            for task in interval:
                cost += self._instance.p[task][operator]
            if cost < maxCost:
                maxCost = cost
        return maxCost

    def _evaluatePopulation(self, population):
        for individual in population:
            individual.fitness = self._getFitness(individual)

    @staticmethod
    def _getFittestIndividual(population):
        return min(population, key=lambda individual: individual.fitness)

    @staticmethod
    def _crossover(parent1, parent2):
        return Individual(parent1.permutation, parent2.intervals), Individual(parent2.permutation, parent1.intervals)

    @staticmethod
    def _mutation(individual):
        def permutation():
            for i in range(rd.randint(1, 10)):
                maxOpId = len(individual.permutation)-1
                op1, op2 = rd.randint(0, maxOpId), rd.randint(0, maxOpId)
                op1_i = individual.permutation.index(op1)
                op2_i = individual.permutation.index(op2)
                individual.permutation[op1_i], individual.permutation[op2_i] = individual.permutation[op2_i], \
                                                                               individual.permutation[op1_i]

        def intervals():
            mutationPoint = rd.randint(1, len(individual.permutation)-1)
            firstOrSecond = rd.randint(0, 1)
            maximumMoveSize = len(individual.intervals[mutationPoint - firstOrSecond])
            if maximumMoveSize > 1:
                maximumMoveSize = rd.randint(1, maximumMoveSize)
                for i in range(maximumMoveSize):
                    if firstOrSecond == 0:
                        task = individual.intervals[mutationPoint][0]
                        individual.intervals[mutationPoint].remove(task)
                        individual.intervals[mutationPoint-1].append(task)
                    else:
                        task = individual.intervals[mutationPoint - 1][-1]
                        individual.intervals[mutationPoint-1].remove(task)
                        individual.intervals[mutationPoint].insert(0, task)

        if len(individual.intervals) > 1:
            permutation()
            intervals()

        return individual

    def evolve(self):
        population = self._generateInitialPopulation()
        self._evaluatePopulation(population)

        bestIndividualOverall = self._getFittestIndividual(population)
        self._stats.firstGenerationObjValue = bestIndividualOverall.fitness

        generationsWithoutImprovement = 0
        self._stats.numberOfGenerations = 1

        prevPop = []
        while generationsWithoutImprovement < self._omega or prevPop == population:
            prevPop = population
            # Select individuals for crossover
            parents = []
            for i in range(int(self._lambda / 2)):
                parent1 = self._randomTournament(population)
                parent2 = self._randomTournament(population)
                parents.append((parent1, parent2))
            # Crossover
            offspring = []
            for parent1, parent2 in parents:
                child1, child2 = self._crossover(parent1, parent2)
                offspring.extend([child1, child2])
            # Mutation
            for individual in offspring:
                r = rd.random()
                if r < self._phi:
                    self._mutation(individual)
            # Calculate Fitness of the offspring
            self._evaluatePopulation(offspring)
            # Merge populations
            population.extend(offspring)
            # Select individuals to pass to the next generation
            population = self._selectNewPopulation(population)
            # Update solution
            currentBestIndividual = self._getFittestIndividual(population)
            if currentBestIndividual.fitness < bestIndividualOverall.fitness:
                bestIndividualOverall = currentBestIndividual
                generationsWithoutImprovement = 0
            else:
                generationsWithoutImprovement += 1
            self._stats.numberOfGenerations += 1
        self._stats.objValue = bestIndividualOverall.fitness
        self._bestIndividual = bestIndividualOverall

    def getStatistics(self):
        return self._stats

    def getIntervals(self):
        return self._bestIndividual.intervals

    def getOperatorsPermutation(self):
        return self._bestIndividual.permutation


def tests():
    inst = str(sys.argv[1])
    M = int(sys.argv[2])
    L = int(sys.argv[3])
    K = int(sys.argv[4])
    P = float(sys.argv[5])
    O = int(sys.argv[6])
    seed = int(sys.argv[7])
    outputfile = str(sys.argv[8])

    rd.seed(seed)

    start = time.time()
    ga = GeneticAlgorithm(inst, M, L, K, P, O)
    ga.evolve()
    end = time.time()

    elapsedTime = end - start
    generations = ga._stats.numberOfGenerations
    firstGeneration = ga._stats.firstGenerationObjValue
    bestGeneration = ga._stats.objValue

    with open(outputfile + ".txt", "a") as output:
        output.write(
            "{0},{1},{2},{3},{4},{5},{6:.2f},{7:.2f},{8:.2f},{9:.2f}\n".format(inst, M, L, K, P, O, elapsedTime,
                                                                               generations,
                                                                               firstGeneration,
                                                                               bestGeneration))


def debug():
    rd.seed(1234554321)
    ga = GeneticAlgorithm("tba1", 1000, 500, 3, 0.5, 500)
    ga.evolve()
    print(ga.getStatistics().objValue)
    print(ga.getOperatorsPermutation())
    print(ga.getIntervals())


if __name__ == "__main__":
    debug()

