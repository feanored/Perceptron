# -*- coding: utf-8 -*-
# neuron.py
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

# Ativar ou desativar o debug dos neurônios
_debug = True

class Neuron:
    def __init__(self, weights, bias, learning_rate, ativacao, der_ativacao):
        '''(list[float], float, float, Callable, Callable) -> None
        Construtor do Neurônio
        '''
        self.weights = weights
        self.bias = bias
        self.ativacao = ativacao
        self.der_ativacao = der_ativacao
        self.learning_rate = learning_rate
        self.output_cache = 0.0
        self.delta = 0.0

    def output(self, inputs):
        '''(list[float]) -> float
        Computa valor de ativação do neurônio, salvando o valor
        antes da função de ativação ser aplicado e retornado
        '''
        self.output_cache = np.dot(inputs, self.weights) + self.bias
        return self.ativacao(self.output_cache)

    def __str__(self):
        '''(None) -> str
        Exibir a saída do neurônio antes da função de ativação
        caso o modo _debug esteja ativado
        '''
        if _debug:
            return "%.9f"%self.output_cache
        else:
            return "**Debug OFF**"

    def __repr__(self):
        '''(None) -> str'''
        return self.__str__()