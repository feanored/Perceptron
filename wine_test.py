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
from dados import DadosWines
from perceptron import Perceptron
#from scores import Scores


def main():

    # Obtendo os dados dos Vinhos a partir da minha classe
    vinhos = DadosWines()
    vinhos.obter_dados()

    # número de dados de treino
    n_train = 138

    # Divide os dados nos conjuntos de treino e teste
    vinhos.split(n_train)
    print(vinhos)

    print("Treinando...")

    # quantidade de treinamentos (padrao=50)
    perceptron = Perceptron(ativacao="sig")

    corretos, acuracia = perceptron.treinar(vinhos.x_train, vinhos.y_train)

    print("\n%d corretos de %d = %.1f%% de acurácia"%(
            corretos, n_train, acuracia*100))

    print("Camada de saída: ", perceptron.network.layers[-1].neurons)

    # minha classe geradora da matriz de confusão
    #scores = Scores(y_test, y_pred)
    #scores.exibir_grafico()

if __name__ == "__main__":
	main()