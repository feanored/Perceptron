# -*- coding: utf-8 -*-
# layer.py
# From Classic Computer Science Problems in Python Chapter 7
# Copyright 2018 David Kopec
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""
@author: Eduardo Galvani Massino
Número USP: 9318532
"""
from util import get_normal_truncada
from neuron import Neuron
import numpy as np

class Layer:
    def __init__(self, previous_layer, num_neurons, learning_rate,
                 ativacao=None, der_ativacao=None):
        '''(Layer, int, float, Callable, Callable) -> None
        Construtor da Camada de Neurônios
        '''
        self.neurons = np.array([], dtype=np.float64)
        self.previous_layer = previous_layer
        self.output_cache = np.zeros(num_neurons)
        # gerador de va normal truncada
        normal_t = get_normal_truncada(mean=0, low=-1, up=1)

        # inicializa pesos aleatoriamente, exceto para camada de entrada
        for i in range(num_neurons):
            pesos = None
            bias = None
            if previous_layer is not None:
                pesos = normal_t.rvs(len(previous_layer.neurons))
                bias = 0.01
            
            neuron = Neuron(pesos, bias, learning_rate, ativacao, der_ativacao)
            self.neurons = np.append(self.neurons, neuron)

    def outputs(self, inputs):
        '''(list[float]) -> list[float]
        Armazena em cache as saidas dos neuronios e a retornam
        Se for uma camada de entrada, usa elas diretamente
        '''
        if self.previous_layer is None:
            self.output_cache = inputs
        else:
            self.output_cache = np.array([n.output(inputs) for n in self.neurons])
        return self.output_cache

    # deve ser chamado somente na camada de saída
    def calcular_delta_camada_de_saida(self, expected):
        '''(list[float]) -> None'''
        for i, neuron in np.ndenumerate(self.neurons):
            der_cost = expected[i[0]] - self.output_cache[i]
            neuron.delta = neuron.der_ativacao(neuron.output_cache) * der_cost

    # deve ser chamado apenas nas camadas ocultas
    def calcular_delta_camada_oculta(self, next_layer):
        '''(Layer) -> None'''
        for i, neuron in np.ndenumerate(self.neurons):
            next_weights = np.array([n.weights[i[0]] for n in next_layer.neurons])
            next_deltas = np.array([n.delta for n in next_layer.neurons])
            der_cost = np.dot(next_weights, next_deltas)
            neuron.delta = neuron.der_ativacao(neuron.output_cache) * der_cost

