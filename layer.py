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
from random import random
from neuron import Neuron
from util import dot_product

class Layer:
    def __init__(self, previous_layer, num_neurons, learning_rate,
                 ativacao, der_ativacao):
        '''(Layer, int, float, Callable, Callable) -> None
        Construtor da Camada de Neurônios
        '''
        self.neurons = []
        self.previous_layer = previous_layer
        # inicializa pesos aleatoriamente, exceto para camada de entrada
        for i in range(num_neurons):
            if previous_layer is None:
                pesos_random = []
            else:
                pesos_random = [random() for _ in range(len(previous_layer.neurons))]
            neuron = Neuron(pesos_random, learning_rate, ativacao, der_ativacao)
            self.neurons.append(neuron)
        self.output_cache = [0.0 for _ in range(num_neurons)]

    def outputs(self, inputs):
        '''(list[float]) -> list[float]
        Armazena em cache as saidas dos neuronios e a retornam
        Se for uma camada de entrada, usa elas diretamente
        '''
        if self.previous_layer is None:
            self.output_cache = inputs
        else:
            self.output_cache = [n.output(inputs) for n in self.neurons]
        return self.output_cache

    # deve ser chamado somente na camada de saída
    def calculate_deltas_for_output_layer(self, expected):
        '''(list[float]) -> None'''
        for n in range(len(self.neurons)):
            self.neurons[n].delta = self.neurons[n].der_ativacao(
                    self.neurons[n].output_cache) * (expected[n] - self.output_cache[n])

    # deve ser chamado apenas nas camadas ocultas
    def calculate_deltas_for_hidden_layer(self, next_layer):
        '''(Layer) -> None'''
        for index, neuron in enumerate(self.neurons):
            next_weights = [n.weights[index] for n in next_layer.neurons]
            next_deltas = [n.delta for n in next_layer.neurons]
            produtos = dot_product(next_weights, next_deltas)
            neuron.delta = neuron.der_ativacao(neuron.output_cache) * produtos

