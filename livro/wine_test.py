# wine_test.py
# From Classic Computer Science Problems in Python Chapter 7
# Copyright 2018 David Kopec
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import csv
from typing import List
from util import normalize_by_feature_scaling
from network import Network
import random as rd
from tqdm import tqdm
from scores import Scores

#rd.seed(123)

def vinhos(i):
	return "Vinho tipo %d"%(i)

def tipos_de_vinhos(output: List[float]) -> str:
        maximo = max(output)
        for i in range(len(output)):
        	if maximo == output[i]:
        		return vinhos(i+1)

def main():
    wine_parameters: List[List[float]] = []
    wine_classifications: List[List[float]] = []
    wine_species: List[int] = []
    with open('wine.csv', mode='r') as wine_file:
        wines: List = list(csv.reader(wine_file, quoting=csv.QUOTE_NONNUMERIC))
        rd.shuffle(wines) # get our lines of data in random order
        for wine in wines:
            parameters: List[float] = [float(n) for n in wine[1:14]]
            wine_parameters.append(parameters)
            species: int = int(wine[0])
            if species == 1:
                wine_classifications.append([1.0, 0.0, 0.0])
            elif species == 2:
                wine_classifications.append([0.0, 1.0, 0.0])
            else:
                wine_classifications.append([0.0, 0.0, 1.0])
            wine_species.append(vinhos(species))
    normalize_by_feature_scaling(wine_parameters)

    network: Network = Network([13, 7, 3], 0.9)

    # número de dados de treino
    n_train = 120
    # quantidade de treinamentos (padrao=50)
    M = 100

    # train over the first 150 wines 10 times
    wine_trainers: List[List[float]] = wine_parameters[:n_train]
    wine_trainers_corrects: List[List[float]] = wine_classifications[:n_train]
    for _ in tqdm(range(M)):
        network.train(wine_trainers, wine_trainers_corrects)

    # test over the last 28 of the wines in the data set
    x_test: List[List[float]] = wine_parameters[n_train:]
    y_test = wine_species[n_train:]
    #wine_results = network.validate(x_test, y_test, tipos_de_vinhos)
    #print(f"{wine_results[0]} correct of {wine_results[1]} = {wine_results[2] * 100}%")

    y_pred = network.predict(x_test, tipos_de_vinhos)
    results = network.validar(y_test)
    print(f"%d corretos de %d = %.1f%%"%(results[0], len(y_test), results[1]*100))

    # minha classe geradora da matriz de confusão
    scores = Scores(y_test, y_pred)
    scores.exibir_grafico()

if __name__ == "__main__":
	main()