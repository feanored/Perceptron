#!/usr/bin/env python
# coding: utf-8

# Ignorar ERROS do Tensorflow relacionados a GPU inexistente
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)

# TensorFlow e tf.keras
import tensorflow as tf
from tensorflow import keras

# Librariesauxiliares
import numpy as np
from scores import Scores
from sklearn.datasets import fetch_openml

print("Classificação da base MNIST usando a API Keras\n")

# Testando o mnist clássico dos números
print("Obtendo dados...")
mnist = fetch_openml('mnist_784', version=1) # versao 28x28

_N = int(mnist.data.shape[0]*0.8)
x_train, y_train = mnist.data[:_N], mnist.target[:_N].astype(np.uint8)
x_test, y_test = mnist.data[_N:], mnist.target[_N:].astype(np.uint8)

# deixando os vetores nos formatos corretos pra usar no keras
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
x_train = x_train.reshape(-1, 28, 28)
x_test = x_test.reshape(-1, 28, 28)


# 3 camadas do tipo "perceptron" que aqui se chama "Dense" pois 
# significa que as camadas estão totalmente conectadas
# (poderia ser diferente, o que configuraria outro tipo de rede)
# além disso uso a melhor função de ativação a ELU
# muuuuuuuito melhorada em relação à sigmoide
model = tf.keras.Sequential()
layers = tf.keras.layers

model.add(layers.Flatten(input_shape=(28, 28)))
model.add(layers.Dense(512, activation='elu'))
model.add(layers.Dense(256, activation='elu'))
model.add(layers.Dense(128, activation='elu'))
model.add(layers.Dense(10, activation='softmax'))

# assim o tipo de rede é definido pelo tipo da conexão dos neurônios
# e o mesmo tipo pode ser treinado de formas diferentes, mas 
# mesmo assim usando esse formato de conexão

# "adam" é outro tipo de otimização, diferente do SGD
model.compile(optimizer=keras.optimizers.SGD(lr=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


print("\nTreinando...\n")
model.fit(x_train, y_train, epochs=15)

# avaliando dados de treinamento
print("\navaliando dados de treinamento")
test_loss, test_acc = model.evaluate(x_train, y_train, verbose=2)

# avaliando os dados de teste
print("\navaliando os dados de teste")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)


# prevendo a partir dos dados de teste
y_pred = np.argmax(model.predict(x_test), axis=-1)

# recuperando as 10 classes de números
# a partir das probabilidades obtidas 
# na camada de saída
'''y_pred = []
for y in preditions:
    y_max = np.argmax(y)
    y_pred.append([y_max])
y_pred = np.array(y_pred).reshape(-1, 1)'''

# usando a minha propria classe de validação
# e que mostra a acuracia e a matriz de confusão
score = Scores(y_test, y_pred)
score.exibir_grafico("Dados de teste")

