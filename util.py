# -*- coding: utf-8 -*-
# util.py
# From Classic Computer Science Problems in Python Chapter 7
# Copyright 2018 David Kopec
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""
@author: Eduardo Galvani Massino
Número USP: 9318532
"""
import numpy as np
#from functools import lru_cache

# produto escalar entre 2 vetores do R^n
def dot_product(xs, ys):
    return sum(x * y for x, y in zip(xs, ys))

# função de ativação Smooth ReLU
# a sua derivada é a função Sigmoide
# https://adl1995.github.io/an-overview-of-activation-functions-used-in-neural-networks.html
# https://towardsdatascience.com/activation-functions-and-its-types-which-is-better-a9a5310cc8f
#@lru_cache(maxsize=None)
def s_relu(x):
    '''(float) -> float'''
    try:
        return np.log(1 + np.exp(x))
    except OverflowError:
        print(x)
        raise OverflowError("Valor estranho")
'''
O que aprendi até agora:
    Com a smooth relu, o algoritmo performa melhor
    quanto menor é a taxa de aprendizado e menor a quantidade
    de neurônios, até o limite do número de neurônios da camada
    da saída, o que parece lógico em algum sentido.
'''

# função sigmóide, usada para classificação
#@lru_cache(maxsize=None)
def sigmoid(x):
    '''(float) -> float'''
    try:
        return 1.0 / (1.0 + np.exp(-x))
    except OverflowError:
        print(x)
        raise OverflowError("Valor estranho")
'''
O que aprendi até agora:
    Com a sigmoide, o algoritmo performa melhor
    com uma taxa um pouco maior do que aquela da Smooth
    e com mais neurônios, pelo menos o triplo da qtde da saída
'''

# derivada da função sigmóide
#@lru_cache(maxsize=None)
def derivative_sigmoid(x):
    '''(float) -> float'''
    sig = sigmoid(x)
    return sig * (1 - sig)

# Função de ativação linear
def idem(x):
    return x

# Derivada constante da ativação linear
def one(x):
    return 1


def normalizar(dados):
    '''(list(list)) -> None
    Faz a normalização dos dados, que ficarão de forma que
    o minimo = 0 e o maximo = 1
    '''
    for col_num in range(len(dados[0])):
        coluna = [row[col_num] for row in dados]
        maximo = max(coluna)
        minimo = min(coluna)
        amplitude = maximo - minimo
        for row_num in range(len(dados)):
            dados[row_num][col_num] = (dados[row_num][col_num] - minimo) / amplitude

