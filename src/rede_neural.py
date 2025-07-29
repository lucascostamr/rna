from random import uniform
from utils import sigmoid, sigmoid_derivative

class RedeNeural:
    def __init__(self, input_size=20, hidden_size=5, taxa=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = taxa

        self.v1k = [[uniform(-1, 1) for _ in range(input_size)] for _ in range(hidden_size)]
        self.w11 = [uniform(-1, 1) for _ in range(hidden_size)]
        self.bias_h = [uniform(-1, 1) for _ in range(hidden_size)]
        self.bias_out = uniform(-1, 1)

    def feedforward(self, xk):
        self.h = []
        for i in range(self.hidden_size):
            soma = sum(self.v1k[i][j] * xk[j] for j in range(self.input_size)) + self.bias_h[i]
            self.h.append(sigmoid(soma))

        soma_out = sum(self.h[i] * self.w11[i] for i in range(self.hidden_size)) + self.bias_out
        self.y = sigmoid(soma_out)
        return self.y

    def backpropagate(self, xk, target):
        saida = self.feedforward(xk)
        erro = target - saida
        delta_saida = erro * sigmoid_derivative(saida)

        deltas_h = []
        for i in range(self.hidden_size):
            deltas_h.append(delta_saida * self.w11[i] * sigmoid_derivative(self.h[i]))

        for i in range(self.hidden_size):
            self.w11[i] += self.learning_rate * delta_saida * self.h[i]
        self.bias_out += self.learning_rate * delta_saida

        for i in range(self.hidden_size):
            for j in range(self.input_size):
                self.v1k[i][j] += self.learning_rate * deltas_h[i] * xk[j]
            self.bias_h[i] += self.learning_rate * deltas_h[i]

        return erro ** 2

    def treinar(self, dataset, epocas=1000):
        for epoca in range(epocas):
            erro_total = 0
            for x, y in dataset:
                erro_total += self.backpropagate(x, y)
            if epoca % 100 == 0:
                print(f"Época {epoca}, erro médio = {erro_total / len(dataset):.4f}")

    def prever(self, xk):
        return 1 if self.feedforward(xk) > 0.5 else 0
