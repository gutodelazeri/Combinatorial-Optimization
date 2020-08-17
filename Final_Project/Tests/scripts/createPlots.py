import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyfit
import csv

optimality = {'tba1': 0.56, 'tba2': 0.52, 'tba3': 0.48}
# values = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

plt.figure(1)
objValuesRegression = {value: 0 for value in values}
timesRegression = {value: 0 for value in values}
for instance in ['tba1', 'tba2', 'tba3']:
    objValues = {value: None for value in values}
    times = {value: None for value in values}
    for mu in values:
        with open('stats_0004.csv', 'r') as file:
            reader = csv.reader(file, delimiter=',')
            for row in reader:
                if row[0] == instance and row[1] == str(mu):
                    objValues[mu] = ((float(row[-1]) - optimality[instance])/optimality[instance])*100
                    times[mu] = float(row[2])
                    objValuesRegression[mu] += objValues[mu]/3
                    timesRegression[mu] += times[mu]/3
    plt.subplot(1, 2, 1)
    plt.plot(list(objValues.keys()), list(objValues.values()), label=instance)
    plt.subplot(1, 2, 2)
    plt.plot(list(times.keys()),  list(times.values()), label=instance)

plt.subplot(1, 2, 1)
x = np.array(list(objValuesRegression.keys()))
y = np.array(list(objValuesRegression.values()))
print(x)
b, m = polyfit(x, y, 1)
plt.plot(x, (b + m * x), '--', label='Linha de Tendência')
plt.locator_params(axis='y', nbins=20)
plt.locator_params(axis='x', nbins=10)
plt.xlabel('Valor de ϕ')
plt.ylabel('Gap para otimalidade (%)')
plt.legend()


plt.subplot(1, 2, 2)
x = np.array(list(timesRegression.keys()))
y = np.array(list(timesRegression.values()))
b, m = polyfit(x, y, 1)
plt.plot(x, b + m * x, '--', label='Linha de Tendência')
plt.locator_params(axis='y', nbins=20)
plt.locator_params(axis='x', nbins=10)
plt.xlabel('Valor de ϕ')
plt.ylabel('Tempo (segundos)')
plt.legend()

plt.tight_layout()
plt.show()
