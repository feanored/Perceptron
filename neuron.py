# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 15:16:30 2020

@author: Eduardo Galvani Massino
Número USP: 9318532
"""
from util import prod_escalar

class Neuron:
    def __init__(self, pesos, taxa_aprendizado, funcao_ativacao, derivada_ativacao):
        '''(list[float], float, Callable, Callable) -> None
        Construtor do Neurônio
        '''
        self.pesos = pesos
        self.taxa_aprendizado = taxa_aprendizado
        self.funcao_ativacao = funcao_ativacao
        self.derivada_ativacao = derivada_ativacao
        self.output_cache = 0.0
        self.delta = 0.0

    def output(self, entradas):
        '''(list[float]) -> float
        Computa valor de ativação do neurônio, salvando o valor
        antes da função de ativação ser aplicado e retornado
        '''
        self.output_cache = prod_escalar(entradas, self.pesos)
        return self.funcao_ativacao(self.output_cache)

    def __str__(self):
        txt = "Pesos: %s"%(str(self.pesos))
        txt += "Saídas: %s"%(self.output_cache)
        txt += "Delta: %s"%(self.delta)
        return txt

