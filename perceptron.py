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

#https://www.asimovinstitute.org/neural-network-zoo/

import enum
class Constantes(enum.Enum):
    LIMITE = 1000

class Perceptron():
    def __init__(self, estrategia="QTDE", N=[1], M=100, acur_min=0.95,
                 mse_max=0.1, ativacao="sigm", taxa=0.1):
        '''(None, str, list[int], int, float, float, str, float) -> None
        Construtor da minha classe Perceptron

        Parâmetros da classe:
            *estrategia: define qual estrategia de planejamento, opções são:
                **"qtde": estratégia padrão, faz o treinamento "N" vezes
                    com uma quantidade fixa de neuronios ocultos e de
                    taxa de aprendizado;
                **"acuracia": tenta atingir o valor do parâmetro "acur_min",
                    que é a porcentagem de saídas corretas, o que não leva
                    em conta o valor da função de erro propriamente dita,
                    para ao atingir o limite estabelecido para o número
                    de tentativas definido internamente em Constantes.LIMITE;
                **"mse": tenta atingir o valor do parâmetro "mse_min" de
                    acordo com a função de erro quadrática utilizada porém,
                    para ao atingir o limite estabelecido para o número
                    de tentativas definido internamente em Constantes.LIMITE;
            *N: quantidade de neurônios da camada oculta, podendo ser
                especificada um vetor de várias camadas ocultas ou apenas
                uma. Porém só será levado em conta na estratégia de QTDE
                de treinamentos, uma vez que nas demais estrategias, apenas
                uma camada oculta é utilizada e sua quantidade é otimizada
                de acordo com a estratégia.
            *M: quantidade de treinamentos desejada, pode ser maior que o
                LIMITE se a estratégia for a de quantidade de treinamentos,
                o valor padrão é 50;
            *acur_min: valor mínimo da acurácia, levado em conta apenas se a
                estratégia utilizada for a de maximizar a acuracia;
            *mse_max: valor máximo do erro quadrático médio, levado em conta
                apenas se a estratégia utilizada for a de minimizar o erro;
            *ativacao: escolha de uma das 2 funções de ativação disponíveis
                para a(s) camada(s) oculta(s). A camada de saída usará sempre
                a Smooth Relu.
            *taxa: taxa de aprendizagem, padrão de 0.1

        O processamento torna disponiveis os resultados e erros nos parâmetros
        da classe, que podem ser chamados externamente.

        '''
        # Vários prints dentro da classe para debug
        self.__DEBUG = False

        # verificação de argumentos
        if estrategia not in ("qtde", "acuracia", "mse"):
            raise ValueError("Informe uma estratégia válida!")

        if ativacao == "sigm":
            self.ativacao = sigmoid
            self.der_ativacao = derivative_sigmoid
        # A única outra opção disponível para ativação é a Smooth Relu
        # cuja derivada é a função sigmóide
        elif ativacao == "relu":
            self.ativacao = s_relu
            self.der_ativacao = sigmoid
        else:
            raise ValueError("Função de ativação incorreta!")

        if len(N) < 1 or N[0] < 1:
            raise ValueError("Número de neurônios ocultos inválido!")

        if M < 1:
            raise ValueError("Número de treinamentos inválido!")

        if mse_max < 0.1:
            raise ValueError("MSE máximo inválido!")

        if acur_min < 0.1 or acur_min > 0.99:
            raise ValueError("Acurácia mínima inválida!")

        if taxa < 0.01 or taxa > 0.99:
            raise ValueError("Taxa de aprendizado inválida!")

        self.estrategia = estrategia
        self._x = []
        self.classes = []
        self.N = N
        self.M = M
        self.acur_min = acur_min
        self.mse_max = mse_max
        self.taxa = taxa

        # Inicializa objeto encoder
        self._enc = OneHotEncoder(handle_unknown='ignore',
                                 categories='auto',
                                 sparse=False)

        # objeto network
        self.network = None

    def reinterpretar_saidas(self, saidas):
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

        for i in range(len(self.N)):
            if self.N[i] < neurons_out:
                self.N[i] = neurons_out

        if self.estrategia == "qtde":
            rede = []
            rede.append(neurons_in)
            for hidden in self.N:
                rede.append(hidden)
            rede.append(neurons_out)

            network = Network(rede, self.taxa, self.ativacao, self.der_ativacao)

            for _ in tqdm(range(self.M)):
                network.train(x_train, y_encoded)

            mse_error = network.mse_error(x_train, y_encoded)

            _ = network.predict(x_train, self.reinterpretar_saidas)
            acuracia = network.validate(y_train)

            print("Acurácia: %.3f"%acuracia, end=" ")
            print("MSE: %.3f"%mse_error)


        if self.estrategia == "acuracia":
            acuracia = 0
            tentativas = 0
            neurons_hidden = neurons_out

            while acuracia < self.acur_min and tentativas < Constantes.LIMITE.value:
                tentativas += 1

                rede = [neurons_in, neurons_hidden, neurons_out]
                network = Network(rede, self.taxa, self.ativacao, self.der_ativacao)

                for _ in tqdm(range(self.M)):
                    network.train(x_train, y_encoded)

                mse_error = network.mse_error(x_train, y_encoded)

                _ = network.predict(x_train, self.reinterpretar_saidas)
                acuracia = network.validate(y_train)

                if self.__DEBUG:
                    print("Acurácia: %.3f"%acuracia, end=" | ")
                    print("MSE: %.3f"%mse_error)
                    print("Parâmetros: Taxa=%.3f | Neuronios=%d\n"
                          %(self.taxa, neurons_hidden))

                if acuracia < self.acur_min:
                    # aumento gradativo da taxa
                    self.taxa += 0.01
                    # aumento gradativo da qtde de neuronios ocultos
                    neurons_hidden += 1

        elif self.estrategia == "mse":
            mse_error = 1
            mse_ant = 1
            acuracia = 0
            acuracia_ant = 0
            tentativas = 0
            network = None
            neurons_hidden = neurons_out

            while mse_error > self.mse_max and mse_ant >= mse_error \
                  and tentativas < Constantes.LIMITE.value:
                tentativas += 1
                network_ant = network
                mse_ant = mse_error
                acuracia_ant = acuracia

                rede = [neurons_in, neurons_hidden, neurons_out]
                network = Network(rede, self.taxa, self.ativacao, self.der_ativacao)

                for _ in tqdm(range(self.M)):
                    network.train(x_train, y_encoded)

                mse_error = network.mse_error(x_train, y_encoded)
                _ = network.predict(x_train, self.reinterpretar_saidas)
                acuracia = network.validate(y_train)

                if self.__DEBUG:
                    print("Acurácia: %.3f"%acuracia, end=" | ")
                    print("MSE: %.3f"%mse_error)
                    print("Parâmetros: Taxa=%.3f | Neuronios=%d\n"
                          %(self.taxa, neurons_hidden))

                if mse_ant >= mse_error:
                    # aumento gradativo da taxa
                    self.taxa += 0.1
                    # aumento gradativo da qtde de neuronios ocultos
                    neurons_hidden += 1
                    # aumento gradativo do número de treinos
                    self.M += 50

            # volta atrás nos parâmetros
            if mse_ant < mse_error:
                network = network_ant
                acuracia = acuracia_ant
                mse_error = mse_ant
                self.taxa -= 0.1
                self.M -= 50

        # salva no objeto a versão final da rede
        self.network = network

        return acuracia, mse_error


    def prever(self, X):
        '''(np.array) -> np.array
        Fazer a previsão de valores, uma vez que a rede esteja treinada
        Vou fazer de forma que retorne o vetor das previsões
        e irá aceitar como entrada qualquer vetor que tenha o
        mesmo shape do que o self._x
        '''
        if self.network is None:
            raise ValueError("O objeto Network não foi inicializado! "+
                             "A função treinar deve ser chamada antes.")
        return self.network.predict(X, self.reinterpretar_saidas)

    def mse_error(self, X, Y):
        '''(np.array, np.array) -> float
        Calcula o erro MSE a partir de saidas obtidas e saidas esperadas
        de acordo com o estado atual da rede, que deve estar treinada
        '''
        if self.network is None:
            raise ValueError("O objeto Network não foi inicializado! "+
                             "A função treinar deve ser chamada antes.")
        y_encoded = self._enc.fit_transform(Y)
        mse_error = self.network.mse_error(X, y_encoded)
        return mse_error


