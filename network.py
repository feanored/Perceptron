# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 17:13:43 2020

@author: Eduardo Galvani Massino
Número USP: 9318532
"""
from functools import reduce
from layer import Layer
from util import s_relu, sigmoide, der_sigmoide

class Network:
    def __init__(self, estrutura, taxa_aprendizado, funcao_ativacao=""):
        '''(list[int], float, str) -> None
        Cria a Rede Perceptron, de acordo com a estrutura desejada
        Sendo que no mínimo espera uma estrutura de 3 camadas,
        sendo 1 camada de entrada, 1 oculta e 1 de saída.
        A estrutura é uma lista de inteiros, contendo a
        qtde de neurônios de cada camada.
        Por padrão estou usando a função de ativação Sigmóide,
        mas as opções disponíveis para funcao_ativacao são:
        "sigmoide" ou "s_relu"
        '''
        if funcao_ativacao in ("", "sigmoide"):
            funcao_ativacao = sigmoide
            derivada_ativacao = der_sigmoide
        # A única outra opção disponível para ativação é a Smooth Relu
        # cuja derivada é a função sigmóide
        else:
            funcao_ativacao = s_relu
            derivada_ativacao = sigmoide

        if len(estrutura) < 3:
            raise ValueError("Erro: deve haver ao menos 3 camadas!")

        # camadas
        self.camadas = []

        # lista de previsoes
        self.previsoes = []

        # camada de entrada
        entrada = Layer(None, estrutura[0], taxa_aprendizado,
                        funcao_ativacao, derivada_ativacao)
        self.camadas.append(entrada)

        # camadas ocultas e de saída
        for n, num_neurons in enumerate(estrutura[1:]):
            #print(n, num_neurons)
            camada = Layer(self.camadas[n], num_neurons, taxa_aprendizado,
                           funcao_ativacao, derivada_ativacao)
            self.camadas.append(camada)


    def outputs(self, entrada):
        '''(list[float]) -> list[float]
        Fornece dados de entrada para a primeira camada, em seguida, a saída
        da primeira é fornecida como entrada para a segunda, a saída da segunda
        para a terceira, e assim por diante. (Loucura do reduce)
        '''
        return reduce(lambda entradas, camada : camada.outputs(entradas),
                      self.camadas, entrada)

    def backpropagate(self, esperado):
        '''(list[float]) -> None
        Calcula as mudanças em cada neurônio com base nos erros da saída
        em comparação com a saída esperada
        '''
        # calcula delta para os neurônios da camada de saída
        ultima_camada = len(self.camadas) - 1
        self.camadas[ultima_camada].calcular_deltas_camada_saida(esperado)

        # calcula delta para as camadas ocultas na ordem inversa
        for l in range(ultima_camada - 1, 0, -1):
            self.camadas[l].calcular_deltas_camada_oculta(self.camadas[l + 1])

        # atualiza os pesos dos neurônios
        for camada in self.camadas[1:]: # ignora camada de entrada
            for neuronio in camada.neuronios:
                for w in range(len(neuronio.pesos)):
                    neuronio.pesos[w] += (neuronio.taxa_aprendizado *
                                      camada.camada_anterior.output_cache[w] *
                                      neuronio.delta)


    def train(self, entradas, esperados):
        '''(list[list[floats]], list[list[floats]]) -> None
        Faz o treino da rede perceptron, passando a lista de amostras
        e seus valores esperados para a função backpropagate
        poder atualizar os pesos (isto configura 1 iteração do treino)
        '''
        for i, x in enumerate(entradas):
            self.outputs(x) # manda para o limbo o y gerado
            y = esperados[i]
            self.backpropagate(y)


    def predict(self, entradas, interpretar):
        '''(list[list[floats]], list[list[floats]], Callable) -> None
        Faz a previsão dos valores da Rede
        '''
        self.previsoes = []
        for entrada in entradas:
            self.previsoes.append(interpretar(self.outputs(entrada)))
        return self.previsoes

    def validate(self, esperados):
        '''(list[list[floats]], list[list[floats]], Callable) -> tuple(int, float)
        Função para validar os exemplos do livro,
        mostrando a matriz de confusão ao final
        '''
        corretos = 0
        for y_pred, esperado in zip(self.previsoes, esperados):
            if y_pred == esperado:
                corretos += 1
        acuracia = corretos / len(self.previsoes)
        return corretos, acuracia







