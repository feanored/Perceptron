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
from scores import Scores

def main():

    print("Obtendo dados...")
    # Obtendo os dados dos Vinhos a partir da minha classe
    vinhos = DadosWines()
    vinhos.obter_dados()

    # número de dados de treino
    n_train = 128

    # Divide os dados nos conjuntos de treino e teste
    vinhos.split(n_train)
    print(vinhos)

    print("Treinando...")

    perceptron = Perceptron(estrategia="mse", ativacao="sigm")

    acuracia, mse = perceptron.treinar(vinhos.x_train, vinhos.y_train)

    '''print("\nAcurácia= %.1f%%"%(acuracia*100))
    print("MSE= %.3f"%(mse))
    print("Taxa=%.3f"%(perceptron.taxa))
    print(perceptron.network.estrutura)'''

    print("\nPrevendo e avaliando...")
    # Fitando os dados de teste
    y_train_pred = perceptron.prever(vinhos.x_train)
    y_test_pred = perceptron.prever(vinhos.x_test)

    # minha classe geradora da matriz de confusão
    scores_t = Scores(vinhos.y_train, y_train_pred)
    scores_t.exibir_grafico("Treino")
    mse = perceptron.mse_error(vinhos.x_train, vinhos.y_train)
    print("Treino MSE= %.3f"%(mse))

    scores = Scores(vinhos.y_test, y_test_pred)
    scores.exibir_grafico("Teste")
    mse = perceptron.mse_error(vinhos.x_test, vinhos.y_test)
    print("Teste MSE= %.3f"%(mse))

    print("\nAté mais!")

if __name__ == "__main__":
	main()