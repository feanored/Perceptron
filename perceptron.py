# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 18:37:12 2020

@author: Eduardo Galvani Massino
Número USP: 9318532
"""
from network import Network

class Perceptron(Network):
	def __init__(self, camada_oculta, taxa_aprendizado=0.1):

		estrutura = [2] # entrada
		estrutura.append(x for x in camada_oculta)
		estrutura.append(5) # saida

		super.__init__(estrutura, taxa_aprendizado, "sigmoide")

	def treinar(self):
		pass

	def processar(self):
		pass

	def medir_pontuacao(self, esperados):

		score = Score(esperados, y_pred)
		score.exibir_grafico()
