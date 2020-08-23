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
import numpy as np
from tqdm import tqdm
from util import normalizar
from perceptron import Perceptron
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
    iris_species = []

    with open('datasets/iris.csv', mode='r') as iris_file:
        irises = list(csv.reader(iris_file))
        rd.shuffle(irises) # get our lines of data in random order
        for iris in irises:
            parameters = [float(n) for n in iris[0:4]]
            iris_parameters.append(parameters)
            species = iris[4]
            iris_species.append(species)
    
    iris_parameters = np.array(iris_parameters)
    normalizar(iris_parameters)
    iris_species = np.array(iris_species).reshape(-1, 1)

    print("\nTreinando...\n")

    iris_network = Perceptron(taxa=0.2, ativacao="l_relu", N=[4], debug=1)

    # número de dados de treino
    n_train = 120

    # train over the first 140 irises in the data set 50 times
    x_train = iris_parameters[:n_train]
    y_train = iris_species[:n_train]
    
    iris_network.treinar(x_train, y_train, M=100)
    y_train_pred = iris_network.prever(x_train)
    scores = Scores(y_train, y_train_pred)
    scores.exibir_grafico()

    # test over the last 10 of the irises in the data set
    x_test = iris_parameters[n_train:]
    y_test = iris_species[n_train:]
    y_test_pred = iris_network.prever(x_test)

    # minha classe geradora da matriz de confusão
    scores = Scores(y_test, y_test_pred)
    scores.exibir_grafico()

if __name__ == "__main__":
    main()
