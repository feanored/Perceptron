# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 18:37:12 2020

@author: Eduardo Galvani Massino
Número USP: 9318532
"""
import numpy as np
from util import *
from tqdm import tqdm
from network import Network
from sklearn.preprocessing import OneHotEncoder

import enum
class Constantes(enum.Enum):
    LIMITE = 10

class Perceptron():
    def __init__(self, estrategia="qtde", N=[1], M=100, acur_min=0.95,
                 mse_max=0.1, ativacao="sigm", taxa=0.1, debug=0):
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
                o valor padrão é 100;
            *acur_min: valor mínimo da acurácia, levado em conta apenas se a
                estratégia utilizada for a de maximizar a acuracia;
            *mse_max: valor máximo do erro quadrático médio, levado em conta
                apenas se a estratégia utilizada for a de minimizar o erro;
            *ativacao: escolha de uma das 2 funções de ativação disponíveis
                para a(s) camada(s) oculta(s). A camada de saída usará sempre
                a Smooth Relu.
            *taxa: taxa de aprendizagem, padrão de 0.1
            *debug: flag para exibição de parâmetros durante o treinamento
        '''
        # Vários prints dentro da classe para debug
        self.__DEBUG = debug
        
        # Objeto tqdm para mostrar progresso do treino
        self.tqdm = tqdm

        # verificação de argumentos
        if estrategia not in ("qtde", "acuracia", "mse"):
            raise ValueError("Informe uma estratégia válida!")

        if ativacao == "linear":
            self.ativacao = idem
            self.der_ativacao = one
        elif ativacao == "sigm":
            self.ativacao = sigmoid
            self.der_ativacao = derivative_sigmoid
        elif ativacao == "s_relu":
            self.ativacao = s_relu
            self.der_ativacao = sigmoid
        elif ativacao == "tanh":
            self.ativacao = tanh
            self.der_ativacao = der_tanh
        elif ativacao == "relu":
            self.ativacao = relu
            self.der_ativacao = der_relu
        elif ativacao == "l_relu":
            self.ativacao = l_relu
            self.der_ativacao = der_l_relu
        elif ativacao == "elu":
            self.ativacao = elu
            self.der_ativacao = der_elu
        else:
            raise ValueError("Função de ativação incorreta!")
        
        # a principio, a saída usa a mesma função de ativação
        # mas poderá ser alterado no programa principal
        self.ativacao_saida = self.ativacao
        self.der_ativacao_saida = self.der_ativacao

        if len(N) < 1 or N[0] < 1:
            raise ValueError("Número de neurônios ocultos inválido!")

        if M < 1:
            raise ValueError("Número de treinamentos inválido!")

        if mse_max < 0.001:
            raise ValueError("MSE máximo inválido!")

        if acur_min < 0.01 or acur_min > 0.99:
            raise ValueError("Acurácia mínima inválida!")

        if taxa <= 0 or taxa >= 1:
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

    def treinar(self, x_train, y_train, M=0):
        '''(np.array, np.array) -> None
        Processo de treinamento da rede neural
        1- Tratar os dados, obtendo as classes das respostas
        2- Treinar de acordo com alguma precisao de autovalidacao
        3- Armazenar no objeto os vetores com o treinamento
        O processamento torna disponiveis os resultados e erros nos parâmetros
        e métodos da classe, que podem ser chamados externamente.
        '''
        # onehotencoder extrai as classes únicas já ordenadas alfabeticamente
        y_encoded = self._enc.fit_transform(y_train)
        self.classes = self._enc.categories_[0]
        if self.__DEBUG == 2:
            print(self._enc.categories)
            print(y_train[:5], len(y_train)) 
            print(y_encoded[:5], len(y_encoded))

        # vou assumir que os dados estejam normalizados/tratados devidamente
        
        # aceita trocar o número de treinos, caso esteja
        # definido na estratégia de qtde e o parâmetro m > 0
        if self.estrategia == "qtde" and M > 0:
            self.M = M
        
        # define qtde de neuronios de entrada e de saída de acordo
        # com os dados de treino
        neurons_in = x_train.shape[1]
        neurons_out = len(self.classes)

        for i in range(len(self.N)):
            if self.N[i] < neurons_out:
                self.N[i] = min(int(np.ceil(neurons_in*2/3 + neurons_out)), neurons_in)
        if self.__DEBUG == 1:
            print("Neurônios: %s"%self.N)

        if self.estrategia == "qtde":
            rede = []
            rede.append(neurons_in)
            for hidden in self.N:
                rede.append(hidden)
            rede.append(neurons_out)
            
            ativacoes = (self.ativacao, self.der_ativacao, 
                         self.ativacao_saida, self.der_ativacao_saida)
            network = Network(rede, self.taxa, ativacoes)

            for _ in self.tqdm(range(self.M)):
                network.train(x_train, y_encoded)
            
            if self.__DEBUG == 1:
                mse_error = network.mse(x_train, y_encoded)
                _ = network.predict(x_train, self.reinterpretar_saidas)
                acuracia = network.validate(y_train)
                print("Acurácia: %.3f"%acuracia, end=" ")
                print("MSE: %.3f"%mse_error)


        elif self.estrategia == "acuracia":
            acuracia = 0
            tentativas = 0
            neurons_hidden = self.N[0]

            while acuracia < self.acur_min and neurons_hidden <= neurons_in \
                  and tentativas < Constantes.LIMITE.value:
                tentativas += 1

                rede = [neurons_in, neurons_hidden, neurons_out]
                ativacoes = (self.ativacao, self.der_ativacao, 
                         self.ativacao_saida, self.der_ativacao_saida)
                network = Network(rede, self.taxa, ativacoes)

                for _ in self.tqdm(range(self.M)):
                    network.train(x_train, y_encoded)

                mse_error = network.mse(x_train, y_encoded)
                _ = network.predict(x_train, self.reinterpretar_saidas)
                acuracia = network.validate(y_train)

                if self.__DEBUG == 1:
                    print("Acurácia: %.3f"%acuracia, end=" | ")
                    print("MSE: %.3f"%mse_error)
                    print("Taxa=%.3f | Estrutura=%s\n"%(self.taxa, rede))

                if acuracia < self.acur_min:
                    # aumento gradativo da taxa
                    self.taxa += 0.1
                if acuracia < self.acur_min and neurons_hidden < neurons_in:
                    # aumento gradativo da qtde de neuronios ocultos
                    neurons_hidden += int(np.ceil((neurons_in-neurons_out)*0.1))
                    neurons_hidden = min(neurons_hidden, neurons_in)

        elif self.estrategia == "mse":
            mse_error = 1
            mse_ant = 1
            tentativas = 0
            network = None
            neurons_hidden = neurons_out

            while mse_error > self.mse_max and mse_ant >= mse_error \
                  and tentativas < Constantes.LIMITE.value:
                tentativas += 1
                network_ant = network
                mse_ant = mse_error

                rede = [neurons_in, neurons_hidden, neurons_out]
                ativacoes = (self.ativacao, self.der_ativacao, 
                         self.ativacao_saida, self.der_ativacao_saida)
                network = Network(rede, self.taxa, ativacoes)

                for _ in self.tqdm(range(self.M)):
                    network.train(x_train, y_encoded)

                mse_error = network.mse(x_train, y_encoded)
                _ = network.predict(x_train, self.reinterpretar_saidas)
                acuracia = network.validate(y_train)

                if self.__DEBUG == 1:
                    print("Acurácia: %.3f"%acuracia, end=" | ")
                    print("MSE: %.3f"%mse_error)
                    print("Taxa=%.3f | Estrutura=%s\n"%(self.taxa, rede))

                if mse_error > self.mse_max and mse_ant >= mse_error \
                   and tentativas < Constantes.LIMITE.value:
                    # aumento gradativo da taxa
                    self.taxa += 0.1
                    # aumento gradativo da qtde de neuronios ocultos
                    neurons_hidden += int(np.floor((neurons_in-neurons_out)*0.1))
                    neurons_hidden = min(neurons_hidden, neurons_in)
                    # aumento gradativo do número de treinos
                    self.M += 5

            # volta atrás nos parâmetros
            if mse_ant < mse_error:
                network = network_ant
                self.taxa -= 0.1
                self.M -= 5

        # salva no objeto a versão final da rede
        self.network = network
        
    def reinterpretar_saidas(self, saidas):
        '''(array) -> np.array
        Reinterpreta o neurônio de saída com as classes desejadas
        Faz isso extraindo o valor máximo de ativação da camada de saída
        como sendo o neurônio que foi ativado, quando é usada uma função
        de ativação na saída
        '''
        maximo = max(saidas)
        saida = np.array([int(x == maximo) for x in saidas])
        return self._enc.inverse_transform(saida.reshape(1, -1))

    def prever(self, X, interpretar=None):
        '''(np.array, Callable) -> np.array
        Fazer a previsão de valores, uma vez que a rede esteja treinada
        Vou fazer de forma que retorne o vetor das previsões
        e irá aceitar como entrada qualquer vetor que tenha o
        mesmo shape do que o self._x
        Se for passada uma função de ativação alternativa, esta será usada
        pra interpretar os neurônios de saída
        '''
        if self.network is None:
            raise ValueError("O objeto Network não foi inicializado! "+
                             "A função treinar deve ser chamada antes.")
        if interpretar is None:
            return self.network.predict(X, self.reinterpretar_saidas)
        return self.network.predict(X, interpretar)
    
    def processar(self, X):
        '''
        Processa o vetor de entradas X e retorna a saída sem interpretação
        '''
        if self.network is None:
            raise ValueError("O objeto Network não foi inicializado! "+
                             "A função treinar deve ser chamada antes.")
        saidas = []
        for x in X:
            saidas.append(self.network.outputs(x))
        return saidas

    def funcao_erro(self, X, Y):
        '''(np.array, np.array, str) -> float
        Calcula o erro MSE a partir de saidas obtidas e saidas esperadas
        de acordo com o estado atual da rede, que deve estar treinada
        Por padrão retorna o erro de Norma 2, o MSE, que está
        implementado no algoritmo de propagação retrógrada.
        '''
        if self.network is None:
            raise ValueError("O objeto Network não foi inicializado! "+
                             "A função treinar deve ser chamada antes.")
        y_encoded = self._enc.fit_transform(Y)
        return self.network.mse(X, y_encoded)


