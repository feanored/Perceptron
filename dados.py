# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 10:21:32 2020

@author: Eduardo Galvani Massino
Número USP: 9318532
"""
import csv
import os.path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from util import normalizar

''' ------------------------------------------------------------------------'''

class Dados:
    def __init__(self):
        '''(None) -> None'''
        self._x, self._y = None, None
        self.x_train, self.y_train = None, None
        self.x_test, self.y_test = None, None


    def __str__(self):
        '''(None) -> str'''
        if self._x is None:
            string = "Não há dados!"
        elif self.x_train is None:
            string = "Objetos:\n"
            string += "X.Shape: "+str(self._x.shape)+"\n"
            string += "Y.Shape: "+str(self._y.shape)+"\n"
        else:
            string = "Conjunto de Treino:\n"
            string += "X.Shape: "+str(self.x_train.shape)+"\n"
            string += "Y.Shape: "+str(self.y_train.shape)+"\n"
            string += "\nConjunto de Teste:\n"
            string += "X.Shape: "+str(self.x_test.shape)+"\n"
            string += "Y.Shape: "+str(self.y_test.shape)+"\n"
        return string

    def __repr__(self):
        '''(None) -> str'''
        return self.__str__()

    def obter_dados(self, op=""):
        pass

    def split(self, n):
        '''(int) -> None'''
        self.x_train = self._x[:n]
        self.y_train = self._y[:n]
        self.x_test = self._x[n:]
        self.y_test = self._y[n:]

''' ------------------------------------------------------------------------'''

class DadosMnist(Dados):
    def __init__(self):
        super().__init__()

    # obtendo o conjunto de imagens de números escritos à mão
    def obter_dados(self, op=""):
        '''(None) -> None'''

        if self._x is None or self._y is None:
            # salvando da web no PC
            if (not os.path.exists("datasets/mnist_img.csv")) or (
                not os.path.exists("datasets/mnist_num.csv")) or (
                op == "force"):

                from sklearn.datasets import fetch_openml
                dados = fetch_openml('mnist_784', version=1)

                imgs = pd.DataFrame(dados.data)
                imgs.to_csv("datasets/mnist_img.csv", header=None,
                            compression="gzip")

                nums = pd.Series(dados.target, index=None)
                nums.to_csv("datasets/mnist_num.csv", header=None,
                            compression="gzip")

            # se já baixei, apenas retorna
            else:
                imgs = pd.read_csv("datasets/mnist_img.csv", header=None,
                       compression="gzip").drop([0], axis=1)
                nums = pd.read_csv("datasets/mnist_num.csv", header=None,
                       compression="gzip").drop([0], axis=1)
                nums = pd.Series.ravel(nums)

            # salva no objeto os vetores numpy
            self._x = np.array(imgs)
            self._y = np.array(nums)

    def split(self, n=60000):
        '''(int) -> None
        Dataset Mnist possui 60 mil dados de treino e 10 mil de teste
        '''
        super().split(n)

    def get_img(self):
        '''(list, list) -> None
        Função auxiliar para mostrar as imagens
        '''
        j = int(np.random.uniform(0, self.x_train.shape[0]))

        # exibindo o valor correspondente à imagem
        print(self.y_train[j])
        some_digit = self.x_train[j]
        #print("Qtde pixels: ", len(some_digit))

        # a imagem é uma matriz quadrada de intensidade do cinza (0 a 255)
        some_digit_image = some_digit.reshape(28, 28)
        #print(some_digit_image)

        # exibindo a imagem
        plt.imshow(some_digit_image, cmap = mpl.cm.binary,
                   interpolation="nearest")
        plt.axis("off")
        plt.show()

''' ------------------------------------------------------------------------'''

class DadosWines(Dados):
    def __init__(self):
        super().__init__()

    def vinhos(self, i):
        return "Vinho tipo %d"%(i)

    def obter_dados(self):
        '''(str) -> None
        Lê os dados de vinhos do csv, os embaralha
        e faz primeiros tratamentos, como a normalização
        dos dados das características dos vinhos
        '''
        x, y = [], []
        with open('datasets\wine.csv', mode='r') as wine_file:
            wines = list(csv.reader(wine_file, quoting=csv.QUOTE_NONNUMERIC))

            # embaralhando os dados
            np.random.shuffle(wines)

            for wine in wines:
                parameters = [float(n) for n in wine[1:14]]
                x.append(parameters)

                # variável resposta está na primeira coluna
                y.append(self.vinhos(wine[0]))

            # normalizando os dados-parâmetros
            normalizar(x)

        self._x = np.array(x)
        self._y = np.array(y).reshape(-1, 1)

    def split(self, n=125):
        '''(int) -> None
        Dataset Mnist possui 60 mil dados de treino e 10 mil de teste
        '''
        super().split(n)




