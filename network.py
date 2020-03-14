# -*- coding: utf-8 -*-
# network.py
# From Classic Computer Science Problems in Python Chapter 7
# Copyright 2018 David Kopec
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""
@author: Eduardo Galvani Massino
Número USP: 9318532
"""
from functools import reduce
from layer import Layer
from util import sigmoid, derivative_sigmoid, s_relu

class Network:
    def __init__(self, layer_structure, learning_rate, funcao_ativacao=""):
        '''(list[int], float, str) -> None
        Cria a Rede Perceptron, de acordo com a estrutura desejada
        Sendo que no mínimo espera uma estrutura de 3 camadas,
        sendo 1 camada de entrada, 1 oculta e 1 de saída.
        A estrutura é uma lista de inteiros, contendo a
        qtde de neurônios de cada camada.
        Por padrão estou usando a função de ativação Sigmóide,
        mas as opções disponíveis para funcao_ativacao são:
        "sigmoide" ou "s_relu"
        '''
        if funcao_ativacao in ("", "sig", "sigmoide"):
            funcao_ativacao = sigmoid
            derivada_ativacao = derivative_sigmoid
        # A única outra opção disponível para ativação é a Smooth Relu
        # cuja derivada é a função sigmóide
        else:
            funcao_ativacao = s_relu
            derivada_ativacao = sigmoid

        if len(layer_structure) < 3:
            raise ValueError("Erro: deve haver ao menos 3 camadas!")

        self.layers = []
        self.previsoes = []
        # input layer
        input_layer: Layer = Layer(None, layer_structure[0], learning_rate,
                                   funcao_ativacao, derivada_ativacao)
        self.layers.append(input_layer)
        # hidden layers and output layer
        for previous, num_neurons in enumerate(layer_structure[1::]):
            next_layer = Layer(self.layers[previous], num_neurons, learning_rate,
                               funcao_ativacao, derivada_ativacao)
            self.layers.append(next_layer)


    def outputs(self, entrada):
        '''(list[float]) -> list[float]
        Fornece dados de entrada para a primeira camada, em seguida, a saída
        da primeira é fornecida como entrada para a segunda, a saída da segunda
        para a terceira, e assim por diante. (Loucura do reduce)
        '''
        return reduce(lambda inputs, layer: layer.outputs(inputs), self.layers, entrada)


    def backpropagate(self, expected):
        '''(list[float]) -> None
        Calcula as mudanças em cada neurônio com base nos erros da saída
        em comparação com a saída esperada
        '''
        # calcula delta para os neurônios da camada de saída
        last_layer: int = len(self.layers) - 1
        self.layers[last_layer].calculate_deltas_for_output_layer(expected)
        # calcula delta para as camadas ocultas na ordem inversa
        for l in range(last_layer - 1, 0, -1):
            self.layers[l].calculate_deltas_for_hidden_layer(self.layers[l + 1])

    def update_weights(self):
        '''(None) -> None
        backpropagate() não atualiza sozinha os pesos
        atualiza os pesos dos neurônios
        '''
        for layer in self.layers[1:]: # pula a camada de entrada
            for neuron in layer.neurons:
                for w in range(len(neuron.weights)):
                    neuron.weights[w] = neuron.weights[w] + (neuron.learning_rate
                         * (layer.previous_layer.output_cache[w]) * neuron.delta)


    def train(self, inputs, expecteds):
        '''(list[list[floats]], list[list[floats]]) -> None
        Faz o treino da rede perceptron, passando a lista de amostras
        e seus valores esperados para a função backpropagate
        poder atualizar os pesos (isto configura 1 iteração do treino)
        '''
        for location, xs in enumerate(inputs):
            ys = expecteds[location]
            _ = self.outputs(xs)
            self.backpropagate(ys)
            self.update_weights()

    def predict(self, entradas, interpretar):
        '''(list[list[floats]], list[list[floats]], Callable) -> None
        Faz a previsão dos valores da Rede
        '''
        self.previsoes = []
        for entrada in entradas:
            self.previsoes.append(interpretar(self.outputs(entrada)))
        return self.previsoes

    def validate(self, esperados):
        '''(list[list[floats]], list[list[floats]], Callable) -> tuple(int, float)
        Função para validar os exemplos do livro,
        mostrando a matriz de confusão ao final.
        DEVE ser chamado após a função predict
        '''
        if len(self.previsoes) == 0:
            raise ValueError("Erro: não há previsões!"+
                             "O método predict deve ser chamado antes!")
        corretos = 0
        for y_pred, esperado in zip(self.previsoes, esperados):
            if y_pred == esperado:
                corretos += 1
        acuracia = corretos / len(self.previsoes)
        return corretos, acuracia
