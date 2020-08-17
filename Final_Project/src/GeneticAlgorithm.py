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
    def __init__(self, instanceName, Mu, Lambda, Phi, Omega, verbose=True):
        self._instance = Instance(instanceName)
        self._mu = Mu
        self._lambda = Lambda
        self._phi = Phi
        self._omega = Omega
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

    def _generateInitialPopulationRandom(self):
        def createIntervals():  # todo: deal with the cases when |m-n| is small and when m > n
            delimiters = []
            intervals = []
            while len(delimiters) < self._instance.m - 1:
                n = rd.randint(0, self._instance.n - 2)
                while n in delimiters:
                    n = rd.randint(0, self._instance.n - 2)
                delimiters.append(n)
            delimiters.sort()
            start = 0
            for delimiter in delimiters:
                intervals.append([i for i in range(start, delimiter + 1)])
                start = delimiter + 1
            intervals.append([i for i in range(start, self._instance.n)])
            return intervals

        population = []
        permutation = [i for i in range(self._instance.m)]
        for k in range(self._mu):
            intervals = createIntervals()
            if len(intervals) > self._instance.m:
                print("oi")
                exit(1)
            rd.shuffle(permutation)
            population.append(Individual(permutation.copy(), intervals))
        return population

    def _randomTournament(self, population):
        if len(population) / 2 >= 3:
            sample = []
            for _ in range(3):
                index = rd.randint(0, self._mu - 1)
                while population[index] in sample:
                    index = rd.randint(0, self._mu - 1)
                sample.append(population[index])
            return self._getFittestIndividual(sample)
        else:
            sample = population.copy()
            sampleSize = len(sample)
            deleted = 0
            while deleted < 3:
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

    @staticmethod
    def _mutation(individual):
        def permutation():
            for i in range(rd.randint(1, 10)):
                maxOpId = len(individual.permutation) - 1
                op1, op2 = rd.randint(0, maxOpId), rd.randint(0, maxOpId)
                op1_i = individual.permutation.index(op1)
                op2_i = individual.permutation.index(op2)
                individual.permutation[op1_i], individual.permutation[op2_i] = individual.permutation[op2_i], \
                                                                               individual.permutation[op1_i]

        def intervals():
            mutationPoint = rd.randint(1, len(individual.permutation) - 1)
            firstOrSecond = rd.randint(0, 1)
            maximumMoveSize = len(individual.intervals[mutationPoint - firstOrSecond])
            if maximumMoveSize > 1:
                maximumMoveSize = rd.randint(1, maximumMoveSize - 1)
                for i in range(maximumMoveSize):
                    if firstOrSecond == 0:
                        task = individual.intervals[mutationPoint][0]
                        individual.intervals[mutationPoint].remove(task)
                        individual.intervals[mutationPoint - 1].append(task)
                    else:
                        task = individual.intervals[mutationPoint - 1][-1]
                        individual.intervals[mutationPoint - 1].remove(task)
                        individual.intervals[mutationPoint].insert(0, task)

        if len(individual.intervals) > 1:
            permutation()
            intervals()

    def evolve_Simple(self):
        population = self._generateInitialPopulationRandom()
        self._evaluatePopulation(population)

        bestIndividualOverall = self._getFittestIndividual(population)
        self._stats.firstGenerationObjValue = bestIndividualOverall.fitness

        generationsWithoutImprovement = 0
        self._stats.numberOfGenerations = 1

        while generationsWithoutImprovement < self._omega:

            offspring = self._generateInitialPopulationRandom()

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

    def evolve_Crossover(self):
        population = self._generateInitialPopulationRandom()
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

    def evolve_CrossoverAndMutation(self):
        startTime = time.time()
        population = self._generateInitialPopulationRandom()
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

            # Mutation
            for individual in offspring:
                if rd.random() < self._phi:
                    self._mutation(individual)

            self._evaluatePopulation(offspring)
            population.extend(offspring)
            population = self._selectNewPopulation(population)
            currentBestIndividual = self._getFittestIndividual(population)

            if currentBestIndividual.fitness < bestIndividualOverall.fitness:
                bestIndividualOverall = currentBestIndividual
                generationsWithoutImprovement = 0
                if self._verbose:
                    print("Generation {0}: {1}".format(self._stats.numberOfGenerations + 1,
                                                       bestIndividualOverall.fitness))
            else:
                generationsWithoutImprovement += 1

            self._stats.numberOfGenerations += 1

        endTime = time.time()
        self._stats.runningTime = endTime - startTime
        self._stats.objValue = bestIndividualOverall.fitness
        self._bestIndividual = bestIndividualOverall

    def getStatistics(self):
        return self._stats

    def getTasksPartition(self):
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
    ga.evolve_CrossoverAndMutation()
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
    rd.seed(1111011111)
    ga = GeneticAlgorithm("tba1", 1000, 1000, 3, 0.7, 500)
    ga.evolve_CrossoverAndMutation()
    st = ga.getStatistics()
    obj = st.objValue
    print(obj)
    print(ga.getOperatorsPermutation())
    print(ga.getTasksPartition())
    print(ga._getFitness(ga._bestIndividual))
    print(st.numberOfGenerations)


if __name__ == "__main__":
    tests()
