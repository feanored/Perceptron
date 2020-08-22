# -*- coding: utf-8 -*-
# util.py
"""
@author: Eduardo Galvani Massino
Número USP: 9318532
"""
import numpy as np
from scipy.stats import truncnorm

# função sigmoid, a ativação mais clássica
def sigmoid(x):
    '''(float) -> float'''
    return 1.0 / (1.0 + np.exp(-x))

# derivada da função sigmóide
def der_sigmoid(x):
    '''(float) -> float'''
    sig = sigmoid(x)
    return sig * (1 - sig)

def tanh(x):
    '''(float) -> float'''
    return 2*sigmoid(2*x) - 1

def der_tanh(x):
    '''(float) -> float'''
    tan = tanh(x)
    return 1 - tan*tan

def relu(x):
    '''(float) -> float'''
    return max(0, x)

def der_relu(x):
    '''(float) -> float'''
    return 1 if x >= 0 else 0

# parâmetro de vazamento
leaky = 1.0/5.5

def l_relu(x):
    '''(float) -> float'''
    return x if x >= 0 else leaky*x

def der_l_relu(x):
    '''(float) -> float'''
    return 1 if x >= 0 else leaky


def elu(x):
    '''(float) -> float'''
    a = 1
    if x >= 0:
        return x
    else:
        return a*(np.exp(x) - 1)

def der_elu(x):
    '''(float) -> float'''
    a = 1
    if x >= 0:
        return 1
    else:
        return elu(x) + a

# Função de ativação linear
def idem(x):
    return x

# Derivada constante da ativação linear
def one(x):
    return 1

# função de ativação Smooth ReLU
# a sua derivada é a função Sigmoid
def s_relu(x):
    '''(float) -> float'''
    return np.log(1 + np.exp(x))


# Função que retorna um gerador truncado da distribuição normal
def get_normal_truncada(mean=0, sd=0.3, low=-1, up=1):
    '''(float, float, float, float) -> Object
    Recebe média, desvio-padrão, limite inferior e limite superior
    para retornar um gerador de números aleatórios seguindo uma
    distribuição normal truncada, com os limites setados pelos parâmetros.
    Uso >> X = get_normal_truncada()
        >> X.rvs(10)
    Retorna 10 amostras para a variável aleatória definida com os
    parâmetros-padrão acima informados.
    '''
    return truncnorm((low-mean)/sd, (up-mean)/sd, loc=mean, scale=sd)


# Testa se o código está sendo executado no Jupyter ou não
def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


# normalização de vetores para o intervalo [0, 1]
def normalizar(dados):
    '''(list(list)) -> None
    Faz a normalização dos dados, que ficarão de forma que
    o minimo = 0 e o maximo = 1.
    Antes disso removo os NAN do vetor
    '''
    if is_notebook():
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm

    for col_num in range(len(dados[0])):
        for row_num in range(len(dados)):
            if np.isnan(dados[row_num][col_num]):
                dados[row_num, col_num] = 0

    for col_num in tqdm(range(len(dados[0]))):
        coluna = [row[col_num] for row in dados]
        maximo = max(coluna)
        minimo = min(coluna)
        amplitude = maximo - minimo
        if amplitude == 0:
            amplitude = 1
        for row_num in range(len(dados)):
            dados[row_num, col_num] = (dados[row_num, col_num] - minimo) / amplitude
