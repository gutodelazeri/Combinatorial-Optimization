import csv


def tests():
    instanceName = ['tba1', 'tba2', 'tba3']
    Value = [str(value) for value in range(100, 1100, 100)]

    for instance in instanceName:
        for value in Value:
            with open('summ_0006.csv', 'r') as file:
                reader = csv.reader(file, delimiter=',')
                time = 0
                generations = 0
                firstGeneration = 0
                bestGeneration = 0
                for row in reader:
                    if row[0] == instance and row[5] == value:
                        time += float(row[6])
                        generations += float(row[7])
                        firstGeneration += float(row[8])
                        bestGeneration += float(row[9])
                time /= 5
                generations /= 5
                firstGeneration /= 5
                bestGeneration /= 5

                f = open('stats_0006.csv', 'a')
                f.write(
                    '{0},{1},{2:.2f},{3:.2f},{4:.2f},{5:.2f}\n'.format(instance, value, time, generations, firstGeneration,
                                                                       bestGeneration))
                f.close()

def final():
    instanceName = ['tba1', 'tba2', 'tba3', 'tba4', 'tba5', 'tba6', 'tba7', 'tba8', 'tba9', 'tba10']
    for instance in instanceName:
        with open('summ_0008.csv', 'r') as file:
            reader = csv.reader(file, delimiter=',')
            time = 0
            generations = 0
            firstGeneration = 0
            bestGeneration = 0
            for row in reader:
                if row[0] == instance:
                    time += float(row[6])
                    generations += float(row[7])
                    firstGeneration += float(row[8])
                    bestGeneration += float(row[9])
            time /= 5
            generations /= 5
            firstGeneration /= 5
            bestGeneration /= 5

            f = open('stats_0008.csv', 'a')
            f.write(
                '{0},{1:.2f},{2:.2f},{3:.2f},{4:.2f}\n'.format(instance, time, generations, firstGeneration,
                                                                   bestGeneration))
            f.close()


final()