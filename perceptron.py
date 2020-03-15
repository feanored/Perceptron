# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 18:37:12 2020

@author: Eduardo Galvani Massino
Número USP: 9318532
"""
from sklearn.preprocessing import OneHotEncoder
from util import sigmoid, derivative_sigmoid, s_relu
from network import Network
from tqdm import tqdm
import numpy as np

class Perceptron():
    def __init__(self, ativacao="sig", taxa=0.1):
        '''(None, float, str) -> None
        Construtor da minha classe Perceptron
        Vou criar os objetos aqui
        '''
        # Inicializa objeto encoder
        self._enc = OneHotEncoder(handle_unknown='ignore',
                                 categories='auto',
                                 sparse=False)
        self._x = []
        self.classes = []
        self.taxa = taxa
        if ativacao in ("", "sig", "sigmoide"):
            self.ativacao = sigmoid
            self.der_ativacao = derivative_sigmoid
        # A única outra opção disponível para ativação é a Smooth Relu
        # cuja derivada é a função sigmóide
        else:
            self.ativacao = s_relu
            self.der_ativacao = sigmoid

        # qtde de atualizações por tentativa de treino
        self.M = 1000

        # objeto network
        self.network = None

    def tipos_de_vinhos(self, saidas):
        '''(array) -> np.array
        Reinterpreta o neurônio de saída com as classes desejadas
        Faz isso extraindo o valor máximo de ativação da camada de saída
        como sendo o neurônio que foi ativado
        '''
        maximo = max(saidas)
        saida = np.array([float(x == maximo) for x in saidas])
        return self._enc.inverse_transform(saida.reshape(1, -1))

    def treinar(self, x_train, y_train):
        '''(np.array, np.array) -> int, float
        Aqui vou fazer o processo de treinamento da rede neural
        1- Tratar os dados, obtendo as classes das respostas
        2- Treinar de acordo com alguma precisao de autovalidacao (futuro)
        3- Armazenar no objeto os vetores com o treinamento
        '''
        # onehotencoder extrai as classes únicas já ordenadas alfabeticamente
        y_encoded = self._enc.fit_transform(y_train)
        self.classes = self._enc.categories_[0]

        # vou assumir que os dados estejam normalizados/tratados devidamente

        # define qtde de neuronios de entrada e de saída de acordo
        # com os dados de treino
        neurons_in = x_train.shape[1]
        neurons_out = len(self.classes)

        # a princípio 1 camada, mas já estou bolando
        # como fazer várias camadas algo como
        # a soma de todas dando o valor mínimo ou
        # proporcional a alguma constante disto
        neurons_hidden = neurons_out

        acuracia = 0
        while acuracia < 0.95:
            rede = [neurons_in, neurons_hidden, neurons_out]

            network = Network(rede, self.taxa, self.ativacao, self.der_ativacao)

            for _ in tqdm(range(self.M)):
                network.train(x_train, y_encoded)

            # será implementado em breve o erro MSE
            _ = network.predict(x_train, self.tipos_de_vinhos)
            results = network.validate(y_train)
            acuracia = results[1]

            print("Acurácia: %.3f"%acuracia, end=" ")
            print("Parâmetros: Taxa=%f | Neuronios=%d\n"%(self.taxa, neurons_hidden))

            # aumento gradativo da taxa
            self.taxa += 0.05
            # aumento gradativo da qtde de neuronios ocultos
            neurons_hidden += 1

        # salva no objeto a versão final da rede
        self.network = network

        return results[0], acuracia



    def prever(self, X):
        '''(np.array) -> np.array
        Fazer a previsão de valores, uma vez que a rede esteja treinada
        Vou fazer de forma que retorne o vetor das previsões
        e irá aceitar como entrada qualquer vetor que tenha o
        mesmo shape do que o self._x
        '''
        pass

