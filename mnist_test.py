#!/usr/bin/env python
# coding: utf-8
import numpy as np
from perceptron import Perceptron
from util import *
from scores import Scores
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm

sns.set()

# obtendo o conjunto de imagens de números escritos
from sklearn.datasets import load_digits # versao 8x8
mnist = load_digits()

_N = int(mnist.data.shape[0]*0.85)
x_train, y_train = mnist.data[:_N], mnist.target[:_N].astype(np.uint8)
x_test, y_test = mnist.data[_N:], mnist.target[_N:].astype(np.uint8)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
normalizar(x_train)
normalizar(x_test)
print("\n\n")

perceptron = Perceptron(taxa=0.001, ativacao="l_relu", N=[48, 24], debug=1)
perceptron.N

print("\nTreinando...\n")
perceptron.treinar(x_train, y_train, M=25)

y_train_pred = perceptron.prever(x_train)
mse = perceptron.funcao_erro(x_train, y_train)
score = Scores(y_train, y_train_pred)
score.exibir_grafico("MSE: %f"%mse)

y_pred = perceptron.prever(x_test)
mse = perceptron.funcao_erro(x_test, y_test)
score = Scores(y_test, y_pred)
score.exibir_grafico("MSE: %f"%mse)

