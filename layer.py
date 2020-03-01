# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 15:51:01 2020

@author: Eduardo Galvani Massino
Número USP: 9318532
"""
from random import random
from neuron import Neuron
from util import dot_product

class Layer:
	def __init__(self, camada_anterior, num_neurons, taxa_aprendizado,
				 funcao_ativacao, derivada_ativacao):
		'''(Layer, int, float, Callable, Callable) -> None
		Construtor da Camada de Neurônios
		'''
		self.camada_anterior = camada_anterior
		self.neuronios = []

		for i in range(num_neurons):
			if camada_anterior is None:
				# sem problema por causa da função
				pesos_random = []
			else:
				pesos_random = [random() for _ in range(len(camada_anterior.neuronios))]

				# só vou criar os neurônios se essa não for a camada de entrada
				neuronio = Neuron(pesos_random, taxa_aprendizado,
							  funcao_ativacao, derivada_ativacao)
				self.neuronios.append(neuronio)

		self.output_cache = [0.0 for _ in range(num_neurons)]

	def outputs(self, entradas):
		'''(list[float]) -> list[float]
		Armazena em cache as saidas dos neuronios e a retornam
		Se for uma camada de entrada, usa elas diretamente
		'''
		if self.camada_anterior is None:
			self.output_cache = entradas
		else:
			self.output_cache = [n.output(entradas) for n in self.neuronios]

		return self.output_cache

	# deve ser chamado somente na camada de saída
	def calcular_deltas_camada_saida(self, saida_esperada):
		'''(list[float]) -> None'''
		for n in range(len(self.neuronios)):
			x = self.neuronios[n].output_cache * (saida_esperada[n] - self.output_cache[n])
			self.neuronios[n].delta = self.neuronios[n].derivada_ativacao(x)

	# não deve ser chamado na camada de saída
	def calcular_deltas_camada_oculta(self, proxima_camada):
		'''(Layer) -> None'''
		for i, neuronio in enumerate(self.neuronios):
			proximos_pesos = [n.pesos[i] for n in proxima_camada.neuronios]
			proximos_deltas = [n.delta for n in proxima_camada.neuronios]
			x = dot_product(proximos_pesos, proximos_deltas)
			neuronio.delta = neuronio.derivada_ativacao(neuronio.output_cache) * x






