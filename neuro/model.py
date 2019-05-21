import numpy as np


class Model:

    def __init__(self, architecture):
        self.gradients = {}
        self.training = True
        self.epsilon = 0.01
        self.architecture = architecture
        self.architecture_size = len(architecture)

    def unpack_cache(self, cache, cache_type):
        cached_W_b, cached_z, cached_A_prev = cache
        # print(cached_W_b)
        cache_list = list()
        if 'z' in cache_type:
            cache_list.append(cached_z)
        if 'a' in cache_type:
            cache_list.append(cached_A_prev)
        if 'w' in cache_type:
            cached_w = cached_W_b[0]
            cache_list.append(cached_w)
        if 'b' in cache_type:
            cached_b = cached_W_b[1]
            cache_list.append(cached_b)
        return cache_list

    def compute_gradients(self, dZ, cache):
        """
        Calculating the gradients of the variables in linear unit

        :param dZ: gradient of cost with respect to the linear unit
        :param cache: cached values from forward propagation of the current layer
        :return: gradients of cost with respect to input from previous layer, weight and bias
        """
        cached_A_prev, cached_W = self.unpack_cache(cache, ['a', 'w'])
        m = cached_A_prev.shape[1]
        dW = (1. / m) * np.dot(dZ, cached_A_prev.T)
        db = (1. / m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(cached_W.T, dZ)

        return dA_prev, dW, db

    def forward(self):
        x = 0
        for layer in range(0, len(self.architecture)):
            x = self.architecture[layer][0](x)
        return x

    def backward(self, target):
        # print('backward')
        L = int((self.architecture_size - 1) / 2)
        L -= 1
        AL = self.architecture[self.architecture_size - 1][2]()     # AL
        dAL = -(np.divide(target, AL+self.epsilon) - np.divide((1 - target), (1+self.epsilon - AL)))

        cache = list()
        cache.append(self.architecture[self.architecture_size - 2][1]())    # W,b
        cache.append(self.architecture[self.architecture_size - 2][2]())    # Z
        cache.append(self.architecture[self.architecture_size - 3][2]())    # AL-1

        dz = self.architecture[self.architecture_size - 1][1](dAL, cache, self.unpack_cache)
        self.gradients['dA' + str(L)], \
            self.gradients['dW' + str(L+1)], \
            self.gradients['db' + str(L+1)] = self.compute_gradients(dz, cache)

        for layer in range(self.architecture_size - 3, 0, -2):
            L -= 1
            # print(layer, L)
            cache = list()
            cache.append(self.architecture[layer - 1][1]())  # W,b
            cache.append(self.architecture[layer - 1][2]())  # Z
            cache.append(self.architecture[layer - 2][2]())  # AL-1

            dz = self.architecture[layer][1](self.gradients['dA' + str(L + 1)], cache, self.unpack_cache)
            self.gradients['dA' + str(L)], \
                self.gradients['dW' + str(L + 1)], \
                self.gradients['db' + str(L + 1)] = self.compute_gradients(dz, cache)

        self.update_parameters()
        # return self.gradients

    def update_parameters(self, lr=0.01):
        L = int(((self.architecture_size - 2) / 2) + 1)
        layer = 1
        for node in range(1, self.architecture_size):
            if self.architecture[node][3] == 'linear':
                self.architecture[node][4](lr*self.gradients['dW'+str(layer)], lr*self.gradients['db'+str(layer)])
                layer += 1

    def cross_entropy_loss(self, A, Y):
        m = Y.shape[1]
        cost = np.squeeze(-np.sum(np.multiply(Y, np.log(A)) + np.multiply((1 - Y), np.log(1 - A))) / m)
        return cost

    def accuracy(self, p, y):
        return np.sum((p == y) / p.shape[1])

    def predict(self, X):
        m = X.shape[0]
        p = np.zeros([1, m])
        self.architecture[0][1](X)
        probas = self.forward()

        for i in range(probas.shape[1]):
            if probas[0, i] > 0.5:
                p[0, i] = 1
            else:
                p[0, i] = 0

        return p