# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 14:50:02 2020

@author: Eduardo Galvani Massino
Número USP: 9318532
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from util import s_relu, sigmoide, der_sigmoide

# criando os vetores com as funções
X = np.linspace(-2.5, 2.51, 200)
Y1 = [s_relu(x) for x in X]
Y2 = [sigmoide(x) for x in X]
Y3 = [der_sigmoide(x) for x in X]

# iniciando a figura
sns.set()
fig = plt.figure(figsize=(12,10))

# mostrando funções de ativação
plt.plot(X, Y1, label="Smooth Relu")
plt.plot(X, Y2, label="Sigmoide")
plt.plot(X, Y3, label="Sigmoide^(1)")

# adicionando legenda
plt.legend()

# exibindo gráfico
plt.show()

# Tutorial dos gráficos
# https://www.datacamp.com/community/tutorials/matplotlib-tutorial-python