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
from layer import Layer
from math import sqrt
import numpy as np

class Network:
    def __init__(self, layer_structure, taxa, ativacoes):
        '''(list[int], float, Tuple[Callable]) -> None
        Cria a Rede Perceptron, de acordo com a estrutura desejada
        Sendo que no mínimo espera uma estrutura de 3 camadas,
        sendo 1 camada de entrada, 1 oculta e 1 de saída.
        A estrutura é uma lista de inteiros, contendo a
        qtde de neurônios de cada camada.
        O parametro ativacoes recebe uma tupla com 4 Callables,
        que representam, em ordem, as funçoes de ativação e a sua derivada
        para a(s) camada(s) oculta(s), e as funçoes de ativação e 
        a sua derivada para a camada de saída.
        '''
        l = len(layer_structure)
        if l < 3:
            raise ValueError("Erro: são necessárias ao menos 3 camadas!")
        if ativacoes is None or len(ativacoes) != 4:
            raise ValueError("Erro: deve definir as funções de ativação!")

        self.layers = np.array([], dtype=np.float64)
        self.estrutura = layer_structure

        # camada de entrada =>
        # não há camada anterior e nem função de ativação
        input_layer = Layer(None, self.estrutura[0], taxa)
        self.layers = np.append(self.layers, input_layer)

        # camadas oculta(s)
        for previous, qtd_neurons in np.ndenumerate(self.estrutura[1::l]):
            next_layer = Layer(self.layers[previous[0]], qtd_neurons, taxa,
                               ativacoes[0], ativacoes[1])
            self.layers = np.append(self.layers, next_layer)

        # camada de saída
        output_layer = Layer(self.layers[-1], self.estrutura[-1], taxa,
                               ativacoes[2], ativacoes[3])
        self.layers = np.append(self.layers, output_layer)


    def feedforward(self, entrada):
        '''(list[float]) -> list[float]
        Fornece dados de entrada para a primeira camada, em seguida, a saída
        da primeira é fornecida como entrada para a segunda, a saída da segunda
        para a terceira, e assim por diante. Ao final, retorna as saidas da camada de saída.
        '''
        saida = self.layers[0].outputs(entrada)
        for i in range(1, len(self.layers)):
            saida = self.layers[i].outputs(saida)
        return saida


    def backpropagate(self, saidas_reais):
        '''(list[float]) -> None
        Calcula as mudanças em cada neurônio com base nos erros da saída
        em comparação com a saída esperada
        '''
        # calcula delta para os neurônios da camada de saída
        last_layer = len(self.layers) - 1
        self.layers[last_layer].calcular_delta_camada_de_saida(saidas_reais)
        
        # calcula delta para as camadas ocultas, da saída para o início da rede
        for l in range(last_layer - 1, 0, -1):
            self.layers[l].calcular_delta_camada_oculta(self.layers[l + 1])

    def update_weights(self):
        '''(None) -> None
        backpropagate() não atualiza sozinha os pesos
        atualiza os pesos dos neurônios
        '''
        for layer in self.layers[1:]: # pula a camada de entrada
            for neuron in layer.neurons:
                for w in range(len(neuron.weights)):
                    neuron.weights[w,] = neuron.weights[w,] + (neuron.learning_rate
                         * (layer.previous_layer.output_cache[w]) * neuron.delta)

    def update_bias(self):
        '''(None) -> None
        backpropagate() não atualiza sozinha o bias
        Atualiza o bias dos neurônios
        '''
        for layer in self.layers[1:]: # pula a camada de entrada
            for neuron in layer.neurons:
                neuron.bias = neuron.bias + neuron.learning_rate * neuron.delta
    
    def train(self, entradas, saidas_reais):
        '''(list[list[floats]], list[list[floats]]) -> None
        Faz o treino da rede perceptron, passando a lista de amostras
        e seus valores esperados para a função backpropagate
        poder atualizar os pesos (isto configura 1 iteração do treino)
        '''
        for i, xs in enumerate(entradas):
            ys = saidas_reais[i]
            _ = self.feedforward(xs)
            self.backpropagate(ys)
            self.update_weights()
            self.update_bias()

    def mse(self, entradas, saidas_reais):
        '''(list[list[floats]], list[list[floats]]) -> float
        Calcula o erro de norma L2 "médio" (MSE)
        '''
        erro = np.array([], dtype=np.float64)
        for j, xs in enumerate(entradas):
            l2 = 0
            ys = saidas_reais[j]
            saidas = self.feedforward(xs)
            for i in range(len(ys)):
                l2 += (ys[i] - saidas[i])**2
            erro = np.append(erro, sqrt(l2))
        return np.mean(erro)

    def predict(self, entradas, interpretar):
        '''(list[list[floats]], list[list[floats]], Callable) -> None
        Faz a previsão dos valores da Rede
        '''
        self.previsoes = np.array([], dtype=np.float64)
        for entrada in entradas:
            self.previsoes = np.append(self.previsoes, interpretar(self.feedforward(entrada)))
        return self.previsoes.reshape(-1, 1)

    def validate(self, esperados):
        '''(list[list[floats]], list[list[floats]], Callable) -> float
        Função que calcula a acurácia da rede para previsões já feitas.
        DEVE ser chamado após a função predict.
        '''
        if self.previsoes is None or len(self.previsoes) == 0:
            raise ValueError("Erro: não há previsões! "+
                             "O método predict deve ser chamado antes!")
        corretos = 0
        for y_pred, esperado in zip(self.previsoes, esperados):
            if y_pred == esperado:
                corretos += 1
        acuracia = corretos / len(self.previsoes)
        return acuracia
