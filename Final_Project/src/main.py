import argparse
from IPSolver import IPSolver as IP
from GeneticAlgorithm import GeneticAlgorithm as GA


def parseInput():
    parser = argparse.ArgumentParser(description='INF05010 - Final Project')
    # todo: outputfile should be optional an have a default value
    parser.add_argument('instanceName', help='Name of the instance')
    parser.add_argument('method', help='Method used to solve the instance. It can be Genetic Algorithm (ga) or '
                                       'Integer Programming (ip)')

    parser.add_argument('-f', '--outputFile', help='Name of the file to store the solution', type=str, default='')
    parser.add_argument("-v", "--Verbose", help="Increase output verbosity", action="store_true")
    parser.add_argument("-t", "--TimeLimit", type=int, default=1800,
                        help="Limit of time in seconds for the method. Should be a positive integer.")
    parser.add_argument("-m", "--Mu", type=int, default=1000, help="Size of the population. Should be a positive integer")
    parser.add_argument("-l", "--Lambda", type=int, default=1000,
                        help="Number of individuals created by crossover. Should be an even positive integer.")
    parser.add_argument("-p", "--Phi", type=float, default=0.25,
                        help="Likelihood of an individual to mutate. Should be a float between 0 and 1.")
    parser.add_argument("-o", "--Omega", type=int, default=500,
                        help="Maximum number of generations without improvement. Should be a positive integer.")

    args = parser.parse_args()

    errors = []
    if args.method != "ga" and args.method != "ip":
        errors.append(
            "Could not recognize the desired method. Acceptable methods are genetic algorithm (ga) or integer "
            "programming (ip).")
    if args.TimeLimit <= 0:
        errors.append("   The time limit value must be a positive integer.")
    if args.Mu <= 0:
        errors.append("   The value of mu must be a positive integer.")
    if args.Lambda <= 0:
        errors.append("   The value of lambda must be a positive integer.")
    if args.Phi < 0 or args.Phi > 1:
        errors.append("   The value of phi must be a float between 0 and 1.")
    if args.Omega <= 0:
        errors.append("   The value of omega must be a positive integer.")

    if len(errors) >= 1:
        print("> The following parameters have invalid values:")
        for e in errors:
            print(e)
        exit(1)
    else:
        return args.outputFile, args.instanceName, args.method, args.Verbose, args.TimeLimit, args.Mu, args.Lambda, args.Phi, args.Omega


def saveSolution(outFile, stats, solution):
    with open(outFile + ".dat", 'w') as out:
        out.write("Instance name: {0}\n".format(stats.instanceName))
        out.write("Method: {0}\n".format(stats.method))
        out.write("Elapsed Time: {0}\n".format(stats.runningTime))
        out.write("Objective Value: {0}\n".format(stats.objValue))
        out.write("-----\n")
        out.write("Permutation: {0}\n".format(solution.getOperatorsPermutation()))
        out.write("Intervals: {0}\n".format(solution.getTasksPartition()))


def printSolution(solution, stats):
    print("-----")
    print("Instance name: {0}".format(stats.instanceName))
    print("Method: {0}".format(stats.method))
    print("Elapsed Time: {0}".format(stats.runningTime))
    print("Objective Value: {0}".format(stats.objValue))
    print("-----")
    print("Permutation: {0}".format(solution.getOperatorsPermutation()))
    print("Intervals: {0}".format(solution.getTasksPartition()))


def main():
    outFile, instance, method, v, t, m, l, p, o = parseInput()
    if method == "ga":
        ga = GA(instance, m, l, p, o, v)
        ga.evolve_CrossoverAndMutation()
        statistics = ga.getStatistics()
        solution = ga
    else:
        ip = IP(instance, t, v)
        ip.solveModel()
        statistics = ip.getStatistics()
        solution = ip
    if outFile != '':
        saveSolution(outFile, statistics, solution)

    printSolution(solution, statistics)


if __name__ == "__main__":
    main()
