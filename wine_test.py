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

    # padronizando nomes
    x_train, x_test = vinhos.x_train, vinhos.x_test
    y_train, y_test = vinhos.y_train, vinhos.y_test

    print("Treinando...")

    perceptron = Perceptron(estrategia="acuracia", ativacao="sigm", mse_max=0.01)
    perceptron.treinar(x_train, y_train)

    print("Estrutura da rede:", perceptron.network.estrutura)
    print("Taxa Ótima= %.3f"%(perceptron.taxa))

    print("\nPrevendo e avaliando...")
    # Obtendo previsões do conjunto de treino (apenas para debug)
    y_train_pred = perceptron.prever(vinhos.x_train)

    # minha classe geradora da matriz de confusão
    scores_t = Scores(vinhos.y_train, y_train_pred)
    scores_t.exibir_grafico("Treino")
    print("MSE (Treino)= %.3f"%(perceptron.funcao_erro(x_train, y_train)))
    print("L1  (Treino)= %.3f"%(perceptron.funcao_erro(x_train, y_train, norma="l1")))

    # Fitando os dados de teste
    y_test_pred = perceptron.prever(x_test)

    # minha classe geradora da matriz de confusão
    scores = Scores(y_test, y_test_pred)
    scores.exibir_grafico("Teste")
    print("MSE (Teste)= %.3f"%(perceptron.funcao_erro(x_test, y_test)))
    print("L1  (Teste)= %.3f"%(perceptron.funcao_erro(x_test, y_test, norma="l1")))

    print("\nAté mais!")

if __name__ == "__main__":
	main()