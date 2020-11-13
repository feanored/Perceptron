import numpy as np

def gerar_janelas(lista, tam):
    '''(np.array, int) -> np.array, np.array'''
    x, y = [], []
    n = len(lista)
    for i in range(n-tam):
        janela = lista[i:i+tam]
        x.append(janela)
        y.append(lista[i+tam])
    return np.array(x), np.array(y)

dados = np.arange(1, 101, 1)
x, y = gerar_janelas(dados, 10)

print(len(dados), "dados =>", len(x), "janelas")
for i in range(len(x)):
    print(x[i], y[i])