# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 19:18:35 2020

@author: Eduardo Galvani Massino
Número USP: 9318532
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from dados import DadosMnist
from scores import Scores
from perceptron import Perceptron

''' ------------------------------------------------------------------------'''

# variavel global para facilitar os testes no console
mnist = None
scores = None

def main():
	'''(None) -> None'''
	global mnist, scores

	# Obter dados dos números
	mnist = DadosMnist()

	print("Obtendo dados... ", end='')
	mnist.obter_dados()
	print("concluído!\n")

	# Separando os dados nos conjuntos de treino e de teste
	mnist.split()

	print(mnist)
	#mnist.get_img()

	# Treinando o modelo
	print("Treinando... ", end='')

	# Testando com modelo pronto
	'''from sklearn.linear_model import SGDClassifier
	sgd = SGDClassifier(random_state=418, loss='perceptron', penalty='l1')
	sgd.fit(mnist.x_train, mnist.y_train)
	y_test_pred = sgd.predict(mnist.x_test)
	'''

	# Usando o meu perceptron
	percep = Perceptron()

	print("concluído!")

	print("Previsão: ", y_test_pred)

	# criei minhas funções para calcular os scores
	print("\nMatriz de confusão:")
	scores = Scores(mnist.y_test, y_test_pred)
	scores.exibir_grafico()
	print("\nAcurácia: %.4f"%scores.acuracia)


''' ------------------------------------------------------------------------'''
if __name__ == "__main__":
	main()