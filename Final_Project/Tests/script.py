import csv

instanceName = ['tba1', 'tba2', 'tba3', 'tba4', 'tba5', 'tba6', 'tba7', 'tba8', 'tba9', 'tba10']
Mu = ['100', '200', '300', '400', '500', '600', '700', '800', '900', '1000']

for instance in instanceName:
    for value in Mu:
        with open('summ_0003.csv', 'r') as file:
            reader = csv.reader(file, delimiter=',')
            time = 0
            generations = 0
            firstGeneration = 0
            bestGeneration = 0
            for row in reader:
                if row[0] == instance and row[1] == value:
                    time += float(row[6])
                    generations += float(row[7])
                    firstGeneration += float(row[8])
                    bestGeneration += float(row[9])
            time /= 3
            generations /= 3
            firstGeneration /= 3
            bestGeneration /= 3

            f = open('0003.csv', 'a')
            f.write(
                '{0},{1},{2:.1f},{3:.1f},{4:.1f},{5:.1f}\n'.format(instance, value, time, generations, firstGeneration,
                                                                   bestGeneration))
            f.close()
