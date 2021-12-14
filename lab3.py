import sys
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

Tk().withdraw()
filename = askopenfilename()
f = open(filename)

fileValues = pd.read_csv(f, delimiter='\s+', header=None, names=["points", "y"])
points = fileValues.iloc[:, [0, 1]].values
N = len(points)
y = np.zeros(N) #заполняем массив нулями

print("Введите количество кластеров:")
variable = int(input())

if variable < 1:
    print("Введите корректное количество кластеров!")
    sys.exit()


def k_means(num_cluster, x_point, y_point, points_number):
    is_last_step = True
    is_first_step = True
    prev_centroid = []
    centroid = []
    avg_arr = []  #массив минимальных расстояний до кластера
    avg = 0
    while is_last_step:
        if is_first_step:
            is_first_step = False
            # Вибираємо випадкове значення для центрів кластерів (центроїд).
            start_point = np.random.choice(range(points_number), num_cluster, replace=False)
            centroid = x_point[start_point]
            # Присваємо кожну xi до найближчого кластеру, обчислюючи відстань до кожного з центроїдів (евклидова метрика)
        else:
            prev_centroid = np.copy(centroid)
            for i in range(num_cluster):
                centroid[i] = np.mean(x_point[y_point == i], axis=0)
                # Знаходимо нове положення центроїдів - як середнє положення точок цього кластера.
        for i in range(points_number):
            dist = np.sum((centroid - x_point[i]) ** 2, axis=1)  # евклидова метрика
            avg_arr.append(min(dist))  # записываем расстояние от точек до центроида
            min_ind = np.argmin(dist)
            y_point[i] = min_ind  # индекс центроиды для каждой точки, к которому находится ближе
        if np.array_equiv(centroid, prev_centroid):  # условия выхода, если центроиды не перестроились,
            # то выходим из цикла
            avg = np.mean(avg_arr)  # среднее отклонение точки от кластера
            is_last_step = False
    avg_arr.clear()
    return avg, x_point, y_point


sse = []   #
means, x, y = k_means(variable, points, y, N)
sse.append(means)  #записываем среднее отклонение точки от кластера
matplotlib.rc('figure', figsize=(10, 10))
for k in range(variable):
    fig = plt.scatter(x[y == k, 0], x[y == k, 1])
plt.show()

for i in range(1, variable):
    means, x, y = k_means(i, x, y, N)
    sse.append(means)

matplotlib.rc('figure', figsize=(10, 5))
sse.sort(reverse=True)
plt.plot(list(range(1, variable + 1)), sse)
plt.xticks(list(range(1, variable + 1)))
plt.scatter(list(range(1, variable + 1)), sse)
plt.xlabel("Amount of Clusters")
plt.ylabel("SSE")
plt.show()
