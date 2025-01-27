import numpy as np

class NeuralNetwork:
    def __init__(self, input_size=784, hidden_size1=40, hidden_size2=20, output_size=10):
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        self.w1, self.b1, self.w2, self.b2, self.w3, self.b3 = self.params()
        
    def params(self):
        w1 = np.random.uniform(-0.5, 0.5, (40, 784))
        w2 = np.random.uniform(-0.5, 0.5, (20, 40))
        w3 = np.random.uniform(-0.5, 0.5, (10, 20))
        b1 = np.zeros((40, 1))
        b2 = np.zeros((20, 1))
        b3 = np.zeros((10, 1))
        return w1, b1, w2, b2, w3, b3

    def ReLU(self, Z):
        return np.maximum(Z, 0)

    def softmax(self, Z_f):
        return np.exp(Z_f) / np.sum(np.exp(Z_f), axis=0, keepdims=True)

    def one_hot(self, L):
        one_hot_L = np.zeros((L.size, L.max() + 1))
        one_hot_L[np.arange(L.size), L] = 1
        return one_hot_L.T

    def deriv_ReLU(self, Z):
        return Z > 0

    def forward_prop(self, X):
        z1 = self.w1.dot(X) + self.b1
        a1 = self.ReLU(z1)
        z2 = self.w2.dot(a1) + self.b2
        a2 = self.ReLU(z2)
        z3 = self.w3.dot(a2) + self.b3
        a3 = self.softmax(z3)
        return z1, a1, z2, a2, z3, a3

    def backward_prop(self, z1, a1, z2, a2, z3, a3, Y, X):
        dz3 = a3 - self.one_hot(Y)
        dw3 = 1/X.shape[1] * dz3.dot(a2.T)
        db3 = 1/X.shape[1] * np.sum(dz3, axis=1, keepdims=True)
        dz2 = self.w3.T.dot(dz3) * self.deriv_ReLU(z2)
        dw2 = 1/X.shape[1] * dz2.dot(a1.T)
        db2 = 1/X.shape[1] * np.sum(dz2, axis=1, keepdims=True)
        dz1 = self.w2.T.dot(dz2) * self.deriv_ReLU(z1)
        dw1 = 1/X.shape[1] * dz1.dot(X.T)
        db1 = 1/X.shape[1] * np.sum(dz1, axis=1, keepdims=True)
        return dw1, db1, dw2, db2, dw3, db3

    def update_params(self, dw1, db1, dw2, db2, dw3, db3, alpha):
        self.w1 -= alpha * dw1
        self.b1 -= alpha * db1
        self.w2 -= alpha * dw2
        self.b2 -= alpha * db2
        self.w3 -= alpha * dw3
        self.b3 -= alpha * db3

    def predict(self, a):
        return np.argmax(a, axis=0)

    def get_accuracy(self, prediction, Y):
        return np.sum(prediction == Y) / Y.size

    def gradient_descent(self, X, Y, iterations, alpha):
        for i in range(iterations):
            z1, a1, z2, a2, z3, a3 = self.forward_prop(X)
            dw1, db1, dw2, db2, dw3, db3 = self.backward_prop(z1, a1, z2, a2, z3, a3, Y, X)
            self.update_params(dw1, db1, dw2, db2, dw3, db3, alpha)
            if i % 10 == 0:
                print(f'Iterations: {i}')
                print('Accuracy: ', self.get_accuracy(self.predict(a3), Y))
        return self.w1, self.b1, self.w2, self.b2, self.w3, self.b3

    def predict_test(self, test, Y):
        _, _, _, _, _, a3 = self.forward_prop(test)
        prediction = self.predict(a3)
        print('Accuracy on Test Data: ', self.get_accuracy(self.predict(a3), Y))