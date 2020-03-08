# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 14:52:31 2020

@author: Eduardo Galvani Massino
Número USP: 9318532
"""
import csv
from util import normalizar
from network import Network

import numpy as np
from tqdm import tqdm
from scores import Scores

np.random.seed(234)

def iris_nomes(output):
    '''(list) -> str'''
    if max(output) == output[0]:
        return "Iris-setosa"
    elif max(output) == output[1]:
        return "Iris-versicolor"
    else:
        return "Iris-virginica"

def main():
    iris_dados = []
    iris_classes = []
    iris_especies = []

    # tratando os dados do CSV
    with open('datasets/iris.csv', mode='r') as iris_file:
        irises = list(csv.reader(iris_file))
        np.random.shuffle(irises) # embaralha os dados de iris
        for iris in irises:
            dados = [float(n) for n in iris[0:4]]
            iris_dados.append(dados)
            species = iris[4]
            if species == "Iris-setosa":
                iris_classes.append([1.0, 0.0, 0.0])
            elif species == "Iris-versicolor":
                iris_classes.append([0.0, 1.0, 0.0])
            else:
                iris_classes.append([0.0, 0.0, 1.0])
            iris_especies.append(species)


    # normaliza os dados no intervalo [0, 1]
    normalizar(iris_dados)
    # número de dados de treino
    n_train = 140
    # quantidade de treinamentos (padrao=50)
    M = 100

    # [4, 6, 3]
    # 4 neurônios na camada de entrada
    # 6 neurônios na camada oculta
    # 3 neurônios na camada de saída
    # Taxa de aprendizado = 0.3
    iris_network = Network([4, 6, 3], 0.3)

    # divive os dados em treino e teste
    x_train = iris_dados[:n_train]
    y_train = iris_classes[:n_train]
    x_test = iris_dados[n_train:]
    y_test = iris_especies[n_train:]

    # faz os treinos
    for i in tqdm(range(M)):
        iris_network.train(x_train, y_train)

    # valida usando os dados restantes
    y_pred = iris_network.predict(x_test, iris_nomes)
    results = iris_network.validate(y_test)
    print(f"%d corretos de %d = %.1f%%"%(results[0], len(y_test), results[1]*100))

    # minha classe geradora da matriz de confusão
    scores = Scores(y_test, y_pred)
    scores.exibir_grafico()

if __name__ == "__main__":
    main()









