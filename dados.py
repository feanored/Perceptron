# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 10:21:32 2020

@author: Eduardo Galvani Massino
Número USP: 9318532
"""
import os.path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

''' ------------------------------------------------------------------------'''

class Dados:
	def __init__(self):
		self._x, self._y = None, None
		self.x_train, self.y_train = None, None
		self.x_test, self.y_test = None, None
		pass

	def __str__(self):
		string = "Conjunto de Treino:\n"
		string += "X.Shape: "+str(self.x_train.shape)+"\n"
		string += "Y.Shape: "+str(self.y_train.shape)+"\n"
		string += "\nConjunto de Teste:\n"
		string += "X.Shape: "+str(self.x_test.shape)+"\n"
		string += "Y.Shape: "+str(self.y_test.shape)+"\n"
		return string

	def __repr__(self):
		return self.__str__()

	def obter_dados(self, op=""):
		pass

	def split(self, n):
		'''(int) -> None'''
		self.x_train = self._x[:n]
		self.y_train = self._y[:n]
		self.x_test = self._x[n:]
		self.y_test = self._y[n:]

''' ------------------------------------------------------------------------'''

class DadosMnist(Dados):
	def __init__(self):
		super().__init__()

	# obtendo o conjunto de imagens de números escritos à mão
	def obter_dados(self, op=""):
		'''(None) -> None'''

		if self._x is None or self._y is None:
			# salvando da web no PC
			if (not os.path.exists("datasets/mnist_img.csv")) or (
				not os.path.exists("datasets/mnist_num.csv")) or (
				op == "force"):

				from sklearn.datasets import fetch_openml
				dados = fetch_openml('mnist_784', version=1)

				imgs = pd.DataFrame(dados.data)
				imgs.to_csv("datasets/mnist_img.csv", header=None,
							compression="gzip")

				nums = pd.Series(dados.target, index=None)
				nums.to_csv("datasets/mnist_num.csv", header=None,
							compression="gzip")

			# se já baixei, apenas retorna
			else:
				imgs = pd.read_csv("datasets/mnist_img.csv", header=None,
					   compression="gzip").drop([0], axis=1)
				nums = pd.read_csv("datasets/mnist_num.csv", header=None,
					   compression="gzip").drop([0], axis=1)
				nums = pd.Series.ravel(nums)

			# salva no objeto os vetores numpy
			self._x =  np.array(imgs)
			self._y = np.array(nums)

	def split(self, n=60000):
		'''(int) -> None
		Dataset Mnist possui 60 mil dados de treino e 10 mil de teste
		'''
		super().split(n)

	def get_img(self):
		'''(list, list) -> None
		Função auxiliar para mostrar as imagens
		'''
		j = int(np.random.uniform(0, self.x_train.shape[0]))

		# exibindo o valor correspondente à imagem
		print(self.y_train[j])
		some_digit = self.x_train[j]
		#print("Qtde pixels: ", len(some_digit))

		# a imagem é uma matriz quadrada de intensidade do cinza (0 a 255)
		some_digit_image = some_digit.reshape(28, 28)
		#print(some_digit_image)

		# exibindo a imagem
		plt.imshow(some_digit_image, cmap = mpl.cm.binary,
				   interpolation="nearest")
		plt.axis("off")
		plt.show()

''' ------------------------------------------------------------------------'''








