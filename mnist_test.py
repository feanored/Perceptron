#!/usr/bin/env python
# coding: utf-8
import numpy as np
from perceptron import Perceptron
from util import *
from scores import Scores
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import seaborn as sns
sns.set()

# obtendo o conjunto de imagens de números escritos
from sklearn.datasets import fetch_openml
from sklearn.datasets import load_digits

print("Classificação da base MNIST usando meu Perceptron\n")

# Testando o mnist clássico dos números
print("Obtendo dados...")
mnist = load_digits() # versao 8x8
#mnist = fetch_openml('mnist_784', version=1) # versao 28x28

_N = int(mnist.data.shape[0]*0.8)
x_train, y_train = mnist.data[:_N], mnist.target[:_N].astype(np.uint8)
x_test, y_test = mnist.data[_N:], mnist.target[_N:].astype(np.uint8)

# deixando os vetores nos formatos corretos pra usar no keras
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
normalizar(x_train)
normalizar(x_test)

perceptron = Perceptron(taxa=0.001, ativacao="elu", N=[48, 24])

print("\nTreinando...\n")
perceptron.treinar(x_train, y_train, M=20)

y_train_pred = perceptron.prever(x_train)
score = Scores(y_train, y_train_pred)
score.exibir_grafico("Dados de treino")
#mse = perceptron.funcao_erro(x_train, y_train)
#print("Mse do treino: %f"%mse)

y_test_pred = perceptron.prever(x_test)
score = Scores(y_test, y_test_pred)
score.exibir_grafico("Dados de teste")
#mse = perceptron.funcao_erro(x_test, y_test)
#print("Mse do teste: %f"%mse)
