# -*- coding: utf-8 -*-
# iris_test.py
# From Classic Computer Science Problems in Python Chapter 7
# Copyright 2018 David Kopec
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""
@author: Eduardo Galvani Massino
Número USP: 9318532
"""
import csv
import random as rd
from tqdm import tqdm
from util import normalizar
from network import Network
from scores import Scores

#rd.seed(123)

def iris_interpret_output(output):
        if max(output) == output[0]:
            return "Iris-setosa"
        elif max(output) == output[1]:
            return "Iris-versicolor"
        else:
            return "Iris-virginica"

def main():
    iris_parameters = []
    iris_classifications = []
    iris_species = []

    with open('iris.csv', mode='r') as iris_file:
        irises = list(csv.reader(iris_file))
        rd.shuffle(irises) # get our lines of data in random order
        for iris in irises:
            parameters = [float(n) for n in iris[0:4]]
            iris_parameters.append(parameters)
            species = iris[4]
            if species == "Iris-setosa":
                iris_classifications.append([1.0, 0.0, 0.0])
            elif species == "Iris-versicolor":
                iris_classifications.append([0.0, 1.0, 0.0])
            else:
                iris_classifications.append([0.0, 0.0, 1.0])
            iris_species.append(species)
    normalizar(iris_parameters)

    print("Treinando...")

    iris_network = Network([4, 6, 3], 0.2)

    # número de dados de treino
    n_train = 120
    # quantidade de treinamentos (padrao=50)
    M = 500

    # train over the first 140 irises in the data set 50 times
    iris_trainers = iris_parameters[:n_train]
    iris_trainers_corrects = iris_classifications[:n_train]
    for _ in tqdm(range(M)):
        iris_network.train(iris_trainers, iris_trainers_corrects)

    # test over the last 10 of the irises in the data set
    iris_testers = iris_parameters[n_train:]
    iris_testers_corrects = iris_species[n_train:]

    y_pred = iris_network.predict(iris_testers, iris_interpret_output)
    results = iris_network.validate(iris_testers_corrects)
    print(f"%d corretos de %d = %.1f%%"%(results[0], len(iris_testers), results[1]*100))

    # minha classe geradora da matriz de confusão
    scores = Scores(iris_testers_corrects, y_pred)
    scores.exibir_grafico()

if __name__ == "__main__":
    main()
