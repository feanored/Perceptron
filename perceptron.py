# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 18:37:12 2020
Modified on Sat Aug 22 17:55:48 2020

@author: Eduardo Galvani Massino
Número USP: 9318532
"""
import numpy as np
from util import *
from network import Network
from sklearn.preprocessing import OneHotEncoder

class Perceptron():
    def __init__(self, N=[1], M=50, ativacao="sigm", taxa=0.1, debug=0):
        '''(None, str, list[int], int, float, float, str, float) -> None
        Construtor da minha classe Perceptron
        Parâmetros da classe:
            *N: quantidade de neurônios da camada oculta, podendo ser
                especificada um vetor de várias camadas ocultas ou apenas
                uma.
            *M: quantidade de treinamentos desejada, denominado de número
                de "épocas" da rede, o valor padrão é 50;
            *ativacao: escolha de uma das funções de ativação disponíveis
                para a(s) camada(s) oculta(s).
            *taxa: taxa de aprendizagem, padrão de 0.1
            *debug: flag para exibição de parâmetros durante o treinamento
        '''
        self.__DEBUG = debug
        
        # importando as funções de ativação e suas derivadas
        if ativacao == "linear":
            self.ativacao = idem
            self.der_ativacao = one
        elif ativacao == "sigm":
            self.ativacao = sigmoid
            self.der_ativacao = der_sigmoid
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

        if taxa <= 0 or taxa >= 1:
            raise ValueError("Taxa de aprendizado inválida!")

        self.N = N
        self.M = M
        self.taxa = taxa
        self.network = None

        # Inicializa objeto encoder
        self._enc = OneHotEncoder(handle_unknown='ignore',
                                 categories='auto',
                                 sparse=False)
        self.classes = None

        # classe auxiliar para exibir o progresso do treinamento
        if is_notebook():
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
        self.tqdm = tqdm


    def treinar(self, x_train, y_train, M=0):
        '''(np.array, np.array) -> None
        Processo de treinamento da rede neural
        1- Tratar os dados, obtendo as classes das respostas
        2- Treinar um número M de épocas
        3- Armazenar no objeto o estado final da rede, 
        com os pesos e vieses ajustados pelo treinamento
        '''
        # onehotencoder extrai as classes únicas já ordenadas alfabeticamente
        y_encoded = self._enc.fit_transform(y_train)
        classes = self._enc.categories_[0]
        if self.__DEBUG == 1:
            print("Classes: ", classes)

        if self.classes is None:
            self.classes = classes
        elif self.classes.all() != classes.all():
            raise ValueError("A lista de valores para o treino contém classes "+
                             "diferentes do que as usadas no treino anterior!")

        if self.__DEBUG == 2:
            print(self._enc.categories)
            print(y_train[:5], len(y_train)) 
            print(y_encoded[:5], len(y_encoded))
        
        # quantidade de neuronios da camada de entrada é 
        # o tamanho de cada vetor da lista de entradas
        neurons_in = x_train.shape[1]

        # define qtde de neuronios de entrada e de saída de acordo
        # com os dados de treino
        if self.network is None:
            neurons_out = len(self.classes)
            for i in range(len(self.N)):
                if len(self.N) == 1 and self.N[0] < neurons_out:
                    self.N[0] = min(int(np.ceil(neurons_in*2/3 + neurons_out)), neurons_in)
            
            rede = []
            rede.append(neurons_in)
            for hidden in self.N:
                rede.append(hidden)
            rede.append(neurons_out)
            
            ativacoes = (self.ativacao, self.der_ativacao, 
                         self.ativacao_saida, self.der_ativacao_saida)

            self.network = Network(np.array(rede), self.taxa, ativacoes)
        
        elif neurons_in != self.network.estrutura[0]:
            raise ValueError("Os valores de entrada devem ter a mesma"+
                             "estrutura que os valores usados no treinamento!")

        if self.__DEBUG == 1:
            print("Neurônios: %s"%self.N)

        # define o número de treinos, caso esteja
        # definido o parâmetro M > 0
        if M > 0:
            self.M = M

        # retropropagação com M épocas
        for _ in self.tqdm(range(self.M)):
            self.network.train(x_train, y_encoded)
        
        # exibe valores de erro e de acurácia
        if self.__DEBUG == 2:
            _ = self.network.predict(x_train, self.reinterpretar_saidas)
            acuracia = self.network.validate(y_train)
            mse_error = self.network.mse(x_train, y_encoded)
            print("Acurácia: %.3f"%acuracia, end=" ")
            print("MSE: %.3f"%mse_error)

        
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
        mesmo shape do que aquele utilizado no treinamento.
        Se for passada uma função de ativação alternativa, esta será usada
        pra interpretar os neurônios de saída
        '''
        if self.network is None:
            raise ValueError("O objeto Network não foi inicializado! "+
                             "A função treinar deve ser chamada antes.")
        elif X.shape[1] != self.network.estrutura[0]:
            raise ValueError("Os valores de entrada devem ter a mesma"+
                             "estrutura que os valores usados no treinamento!")

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
        elif X.shape[1] != self.network.estrutura[0]:
            raise ValueError("Os valores de entrada devem ter a mesma"+
                             "estrutura que os valores usados no treinamento!")

        saidas = []
        for x in X:
            saidas.append(self.network.feedforward(x))
        return np.array(saidas)

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
        elif X.shape[1] != self.network.estrutura[0]:
            raise ValueError("Os valores de entrada devem ter a mesma"+
                             "estrutura que os valores usados no treinamento!")

        y_encoded = self._enc.fit_transform(Y)
        return self.network.mse(X, y_encoded)


