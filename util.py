# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 19:32:18 2020

@author: Eduardo Galvani Massino
Número USP: 9318532
"""
from math import exp, log
import numpy as np
from functools import lru_cache

# produto escalar
def prod_escalar(X, Y):
    '''(list, list) -> float'''
    return np.array(X).dot(Y)


# função sigmóide, usada para classificação
@lru_cache(maxsize=None)
def sigmoide(x):
    '''(float) -> float'''
    if x > int(1e9):
        return 1.0
    elif x < -int(1e9):
        return 0.0
    try:
        return 1.0 / (1.0 + exp(-x))
    except OverflowError:
        return -1.0

# derivada da função sigmóide
@lru_cache(maxsize=None)
def der_sigmoide(x):
    '''(float) -> float'''
    sig = sigmoide(x)
    return sig * (1 - sig)

# função de ativação Smooth ReLU
# https://adl1995.github.io/an-overview-of-activation-functions-used-in-neural-networks.html
@lru_cache(maxsize=None)
def s_relu(x):
    '''(float) -> float'''
    return log(1 + exp(x))
# derivada da Smooth ReLU é justamente a função Sigmoide

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





