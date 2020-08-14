import random as rd
import time
import sys
from itertools import permutations, combinations
from Instance import Instance
from Individual import Individual
from Statistics import Statistics


def createIntervals(baseList, K):
    while True:
        for splits in combinations(range(len(baseList) - 1), K - 1):
            # splits need to be offset by 1, and padded
            splits = [0] + [s + 1 for s in splits] + [None]
            yield [baseList[s:e] for s, e in zip(splits, splits[1:])]


def createPermutations(baseList):
    while True:
        for p in permutations(baseList):
            yield list(p)


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
        self._defaultPermutation = createPermutations([i for i in range(self._instance.m)])
        self._defaultIntervals = createIntervals([i for i in range(self._instance.n)], self._instance.m)

    def _generateInitialPopulation(self, size):
        population = []
        for k in range(size):
            population.append(Individual(next(self._defaultPermutation), next(self._defaultIntervals)))
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
        maxCost = 0
        for interval, operator in zip(individual.intervals, individual.permutation):
            cost = 0
            for task in interval:
                cost += self._instance.p[task][operator]
            if cost > maxCost:
                maxCost = cost
        return maxCost

    def _evaluatePopulation(self, population):
        for individual in population:
            individual.fitness = self._getFitness(individual)

    @staticmethod
    def _getFittestIndividual(population):
        return min(population, key=lambda individual: individual.fitness)

    @staticmethod
    def _crossover1(parent1, parent2):  # described in the work proposal
        return Individual(parent1.permutation, parent2.intervals), Individual(parent2.permutation, parent1.intervals)

    def _crossover2(self, parent1, parent2):  # described
        def permutationCrossover():
            p1 = parent1.permutation
            p2 = parent2.permutation
            maxOpID = len(p1) - 1

            cutPoint1 = rd.randint(1, maxOpID)
            cutPoint2 = rd.randint(1, maxOpID)  # todo: deal with the case when m is small
            while abs(cutPoint1 - cutPoint2) < 2:
                cutPoint2 = rd.randint(1, maxOpID)

            geneCut1 = p1[cutPoint1:cutPoint2]
            geneCut2 = p2[cutPoint1:cutPoint2]
            pool1 = [i for i in p1 if i not in geneCut2]
            pool2 = [i for i in p2 if i not in geneCut1]

            child1 = list(pool2[(maxOpID - cutPoint2 + 1):]) + geneCut1 + list(pool2[:(maxOpID - cutPoint2 + 1)])
            child2 = list(pool1[(maxOpID - cutPoint2 + 1):]) + geneCut2 + list(pool1[:(maxOpID - cutPoint2 + 1)])

            return child1, child2

        def intervalsCrossover():
            numberOfOps = self._instance.m
            numberOfTasks = self._instance.n

            intervals1 = parent1.intervals
            intervals2 = parent2.intervals

            marks_p1 = set([i[0] for i in intervals1])
            marks_p2 = set([i[0] for i in intervals2])
            pool = list(marks_p1.union(marks_p2))

            marks_c1 = {0}
            while len(marks_c1) < numberOfOps:
                marks_c1.add(rd.choice(pool))
            marks_c2 = {0}
            while len(marks_c2) < numberOfOps:
                marks_c2.add(rd.choice(pool))

            marks_c1 = list(marks_c1)
            marks_c1.sort()
            marks_c2 = list(marks_c2)
            marks_c2.sort()

            child1 = []
            child2 = []
            for index in range(numberOfOps):
                if index == numberOfOps - 1:
                    child1.append([e for e in range(marks_c1[index], numberOfTasks)])
                    child2.append([e for e in range(marks_c2[index], numberOfTasks)])
                else:
                    child1.append([e for e in range(marks_c1[index], marks_c1[index + 1])])
                    child2.append([e for e in range(marks_c2[index], marks_c2[index + 1])])

            return child1, child2

        p1, p2 = permutationCrossover()
        i1, i2 = intervalsCrossover()

        return Individual(p1, i1), Individual(p2, i2)  # Order Crossover Operator (0X1) with something else

    def _createGroups(self, population):
        population.sort(key=lambda individual: individual.fitness)
        gr_a = population[:int(self._mu)]
        gr_b = population[int(self._mu):]
        return gr_a, gr_b

    def evolve(self):
        population = self._generateInitialPopulation(self._mu)
        self._evaluatePopulation(population)

        bestIndividualOverall = self._getFittestIndividual(population)
        self._stats.firstGenerationObjValue = bestIndividualOverall.fitness

        generationsWithoutImprovement = 0
        self._stats.numberOfGenerations = 1

        while generationsWithoutImprovement < self._omega:
            # Select individuals for crossover
            parents = []
            for _ in range(int(self._lambda / 2)):
                parent1 = self._randomTournament(population)
                parent2 = self._randomTournament(population)
                parents.append((parent1, parent2))

            # Crossover
            offspring = []
            for parent1, parent2 in parents:
                child1, child2 = self._crossover2(parent1, parent2)
                offspring.extend([child1, child2])

            self._evaluatePopulation(offspring)
            population.extend(offspring)
            population = self._selectNewPopulation(population)
            currentBestIndividual = self._getFittestIndividual(population)

            if currentBestIndividual.fitness < bestIndividualOverall.fitness:
                bestIndividualOverall = currentBestIndividual
                generationsWithoutImprovement = 0
            else:
                generationsWithoutImprovement += 1

            self._stats.numberOfGenerations += 1

        self._stats.objValue = bestIndividualOverall.fitness
        self._bestIndividual = bestIndividualOverall

    def evolve_StructuredPopulation(self):
        population = self._generateInitialPopulation(self._mu)
        self._evaluatePopulation(population)
        A, B = self._createGroups(population)

        bestIndividualOverall = self._getFittestIndividual(A)
        self._stats.firstGenerationObjValue = bestIndividualOverall.fitness

        generationsWithoutImprovement = 0
        self._stats.numberOfGenerations = 1

        elapsedTime = 0
        while generationsWithoutImprovement < self._omega and elapsedTime < 3.6e+12:
            startTime = time.time_ns()
            # Select individuals for crossover
            parents = []
            for _ in range(int(self._mu / 4)):
                parent1 = self._randomTournament(A)
                parent2 = self._randomTournament(A)
                parents.append((parent1, parent2))

            # Crossover
            offspring = []
            for parent1, parent2 in parents:
                child1, child2 = self._crossover2(parent1, parent2)
                offspring.extend([child1, child2])

            # Random
            offspring.extend(self._generateInitialPopulation(int(self._mu/2)))

            self._evaluatePopulation(offspring)
            population.extend(offspring)
            population = self._selectNewPopulation(population)
            A, B = self._createGroups(population)
            currentBestIndividual = self._getFittestIndividual(A)

            if currentBestIndividual.fitness < bestIndividualOverall.fitness:
                bestIndividualOverall = currentBestIndividual
                generationsWithoutImprovement = 0
            else:
                generationsWithoutImprovement += 1

            self._stats.numberOfGenerations += 1
            endTime = time.time_ns()
            elapsedTime += (endTime - startTime)

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
    ga.evolve_StructuredPopulation()
    end = time.time()

    elapsedTime = end - start
    generations = ga._stats.numberOfGenerations
    firstGeneration = ga._stats.firstGenerationObjValue
    bestGeneration = ga._stats.objValue

    with open(outputfile + ".csv", "a") as output:
        output.write(
            "{0},{1},{2},{3},{4},{5},{6:.2f},{7:.2f},{8:.2f},{9:.2f}\n".format(inst, M, L, K, P, O, elapsedTime,
                                                                               generations,
                                                                               firstGeneration,
                                                                               bestGeneration))


def debug():
    rd.seed(10101010101)
    ga = GeneticAlgorithm("tba10", 1000, 1000, 10, 0.7, 500)
    ga.evolve_StructuredPopulation()
    st = ga.getStatistics()
    obj = st.objValue
    print(obj)
    print(ga.getOperatorsPermutation())
    print(ga.getIntervals())
    print(ga._getFitness(ga._bestIndividual))
    print(st.numberOfGenerations)


if __name__ == "__main__":
    tests()
