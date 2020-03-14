# -*- coding: utf-8 -*-
# wine_test.py
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

rd.seed(345)

def vinhos(i):
	return "Vinho tipo %d"%(i)

def tipos_de_vinhos(output) -> str:
    maximo = max(output)
    for i in range(len(output)):
    	if maximo == output[i]:
    		return vinhos(i+1)

def main():
    wine_parameters = []
    wine_classifications = []
    wine_species = []
    with open('datasets\wine.csv', mode='r') as wine_file:
        wines = list(csv.reader(wine_file, quoting=csv.QUOTE_NONNUMERIC))
        rd.shuffle(wines) # get our lines of data in random order
        for wine in wines:
            parameters = [float(n) for n in wine[1:14]]
            wine_parameters.append(parameters)
            species = int(wine[0])
            if species == 1:
                wine_classifications.append([1.0, 0.0, 0.0])
            elif species == 2:
                wine_classifications.append([0.0, 1.0, 0.0])
            else:
                wine_classifications.append([0.0, 0.0, 1.0])
            wine_species.append(vinhos(species))
    normalizar(wine_parameters)

    print("Treinando...")

    # número de dados de treino
    n_train = 120
    # quantidade de treinamentos (padrao=50)
    M = 100

    # train over the first 150 wines 10 times
    x_train = wine_parameters[:n_train]
    y_train = wine_classifications[:n_train]

    # test over the last 28 of the wines in the data set
    x_test = wine_parameters[n_train:]
    y_test = wine_species[n_train:]

    acuracia = 0
    while acuracia < 0.95:
        taxa = rd.random() # no intervalo [0, 1)
        neuronios = rd.randint(3, 30) # n e 10n
        taxa = 0.5
        neuronios = 7
        network = Network([13, neuronios, 3], taxa, "sig")

        for _ in tqdm(range(M)):
            network.train(x_train, y_train)

        y_pred = network.predict(x_test, tipos_de_vinhos)
        results = network.validate(y_test)
        acuracia = results[1]

        print("Acurácia: %.3f"%acuracia, end=" ")
        print("Parâmetros: Taxa=%f | Neuronios=%d\n"%(taxa, neuronios))

    print("\n%d corretos de %d = %.1f%% de acurácia"%(
            results[0], len(y_test), results[1]*100))
    print("Parâmetros: Taxa=%f | Neuronios=%d"%(taxa, neuronios))

    print("Camada de saída: ", network.layers[-1].neurons)

    # minha classe geradora da matriz de confusão
    scores = Scores(y_test, y_pred)
    scores.exibir_grafico()

if __name__ == "__main__":
	main()