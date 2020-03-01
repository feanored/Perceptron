# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 19:18:42 2020

@author: Eduardo Galvani Massino
Número USP: 9318532
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class Scores():
    def __init__(self, x=[], y=[]):
        if len(x)*len(y) > 0 and len(x) == len(y):
            self.x = x
            self.y = y
            self.classes = []
            self._matriz_confusao()
            self._acuracia()
            self._precisao()
            self._recall()
        else:
            raise("Vetores devem existir e ter o mesmo tamanho!")

    def _matriz_confusao(self):
        for x in self.x:
            if x not in self.classes:
                self.classes.append(x)
        self.classes = sorted(self.classes)

        self.matriz = []
        for i in range(len(self.classes)):
            classe_x = self.classes[i]
            linha = []
            for j in range(len(self.classes)):
                classe_y = self.classes[j]
                cont = 0
                for k in range(len(self.x)):
                    if self.x[k] == classe_x and self.y[k] == classe_y:
                        cont += 1
                linha.append(cont)
            self.matriz.append(linha)
        self.matriz = np.array(self.matriz)

    def _acuracia(self): # cálculo da acurácia de uma matriz de confusão (quadrada)
        diag = 0
        for i in range(len(self.matriz)):
            diag += self.matriz[i][i]
        self.acuracia = diag / len(self.x)

    def _precisao(self): # cálculo da precisão de uma matriz de confusão (quadrada)
        if len(self.matriz) == 2:
            self.precisao = self.matriz[1][1] / (self.matriz[0][1] + self.matriz[1][1])
        else:
            self.precisao = self.acuracia

    def _recall(self): # cálculo do recall de uma matriz de confusão (quadrada)
        if len(self.matriz) == 2:
            self.recall = self.matriz[1][1] / (self.matriz[1][0] + self.matriz[1][1])
        else:
            self.recall = self.acuracia

    def exibir_grafico(self):
        '''
        Exibe a matriz de confusão como sendo um gráfico bonito
        '''
        plt.figure(figsize=(8,8))
        sns.set()
        sns.heatmap(self.matriz, square=True, annot=True, fmt='d', cbar=False,
        xticklabels=self.classes,
        yticklabels=self.classes)
        plt.xlabel('Eixo Previsto')
        plt.ylabel('Eixo Real')
        plt.title("Matriz de confusão, acurácia: %.1f%%"%(self.acuracia*100))
        plt.show()