import Instance as inst
import random as rd
# https://stackoverflow.com/questions/7961363/removing-duplicates-in-lists
from collections import OrderedDict


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


class GeneticAlgorithm:
    def __init__(self, instanceName, Mu, Lambda, k, Phi, Omega, timeLimit=1800, verbose=True):
        self._instance = inst.Instance(instanceName)
        self._mu = Mu
        self._lambda = Lambda
        self._k = k
        self._phi = Phi
        self._omega = Omega
        self._timeLimit = timeLimit
        self._verbose = verbose

    # todo: Treat case n < m
    def _generateInitialPopulation(self):
        population = []
        for k in range(self._mu):
            interval = sorted([rd.randint(0, self._instance.m - 1) for _ in range(self._instance.n)])
            permutation = [i for i in range(self._instance.m)]
            rd.shuffle(permutation)
            population.append(Individual(permutation, interval))
        return population

    def _randomTournament(self, population):
        sample = []
        for i in range(self._k):
            index = rd.randint(0, self._mu - 1)
            while population[index] in sample:
                index = rd.randint(0, self._mu - 1)
            sample.append(population[index])

        return self._getFittestIndividual(sample)

    @staticmethod
    def _crossover(parent1, parent2):
        return Individual(parent1.permutation, parent2.interval), Individual(parent2.permutation, parent1.interval)

    def _mutation(self, individual):
        def chooseOperators():
            first = rd.randint(0, self._instance.m - 1)
            second = rd.randint(0, self._instance.m - 1)
            while first == second or first not in individual.interval or second not in individual.interval:
                first = rd.randint(0, self._instance.m - 1)
                second = rd.randint(0, self._instance.m - 1)
            return first, second

        def permutation():
            op1, op2 = chooseOperators()
            op1_i = individual.permutation.index(op1)
            op2_i = individual.permutation.index(op2)
            individual.permutation[op1_i], individual.permutation[op2_i] = individual.permutation[op2_i], \
                                                                           individual.permutation[op1_i]

        def partition():
            increase_op, decrease_op = chooseOperators()
            individual.interval.append(increase_op)  # todo: Maybe the magnitude of these changes can be a parameter
            individual.interval.remove(decrease_op)
            individual.interval.sort()

        permutation()
        partition()

        return individual

    def _selectNewPopulation(self, population):
        population.sort(key=lambda individual: individual.fitness)
        population = population[:-self._lambda]
        return population

    def _getFitness(self, individual):
        costs = [0 for _ in individual.chromosome]
        for task, operator in zip(range(self._instance.n), individual.chromosome):
            costs[operator] += self._instance.p[task][operator]
        return max(costs)

    @staticmethod
    def _getFittestIndividual(population):
        return min(population, key=lambda individual: individual.fitness)

    def _evaluatePopulation(self, population):
        for individual in population:
            individual.generateChromosome()
            individual.fitness = self._getFitness(individual)

    def evolve(self):
        population = self._generateInitialPopulation()
        self._evaluatePopulation(population)
        bestIndividualOverall = self._getFittestIndividual(population)
        generationsWithoutImprovement = 0
        generationsCounter = 0
        print("Generation {0}: {1}".format(generationsCounter, bestIndividualOverall.fitness))
        while generationsWithoutImprovement < self._omega:
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
                if rd.random() < self._phi:
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
            generationsCounter += 1
            print("Generation {0}: {1}".format(generationsCounter, bestIndividualOverall.fitness))

        print(bestIndividualOverall.fitness)


if __name__ == "__main__":
    ga = GeneticAlgorithm("tba0", 20, 4, 4, 0.5, 10)
    ga.evolve()
