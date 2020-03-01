# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 19:32:18 2020

@author: Eduardo Galvani Massino
Número USP: 9318532
"""
from math import exp, log
from functools import lru_cache

# produto escalar
@lru_cache(maxsize=None)
def prod_escalar(X, Y):
	'''(list, list) -> float'''
	return sum(x * y for x, y in zip(X, Y))


# função sigmóide, usada para classificação
@lru_cache(maxsize=None)
def sigmoide(x):
	'''(float) -> float'''
	return 1.0 / (1.0 + exp(-x))

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
