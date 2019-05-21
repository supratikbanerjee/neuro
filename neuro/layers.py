import numpy as np
from scipy.special import expit
from .parameter import Parameter


class Input:
    def __init__(self, x):
        self.x = x

    def forward_features(self, z):
        return self.x.T

    def backward_features(self):
        return self.x.T

    def set_features(self, x):
        self.x = x


class Linear:
    def __init__(self, in_features, out_features, parameter=None, bias=False):
        self.z = np.ndarray
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter([in_features, out_features], parameter)
        if bias:
            self.bias = Parameter([1, out_features], parameter=parameter)
        else:
            self.bias = Parameter([1, out_features], parameter="Zero")

    def forward(self, x):
        self.z = np.add(np.dot(self.weight, x), self.bias)
        self.z = self.z.astype('float32')
        # print(self.z.shape)
        return self.z

    def get_parameters(self):
        return [self.weight, self.bias]

    def get_linear(self):
        return self.z

    def set_parameters(self, W, b):
        self.weight -= W
        self.bias -= b


class ReLU:
    def __init__(self):
        self.dz = np.ndarray
        self.activation = np.ndarray

    def forward(self, x):
        self.activation = np.maximum(0, x)
        self.activation = self.activation.astype('float32')
        return self.activation

    def backward(self, da, cache, unpack_cache):
        cached_z = unpack_cache(cache, ['z'])
        self.dz = np.array(da, copy=True)
        self.dz[cached_z[0] < 0] = 0
        return self.dz

    def get_activation(self):
        return self.activation


class Sigmoid:
    def __init__(self):
        self.dz = np.ndarray
        self.activation = np.ndarray

    def forward(self, x):

        self.activation = expit(x)  # .5 * (1 + np.tanh(.5 * x))    # 1.0 / (1.0 + np.exp(-x))
        self.activation = self.activation.astype('float32')
        return self.activation

    def backward(self, da, cache, unpack_cache):
        cached_z = unpack_cache(cache, ['z'])
        s = self.forward(cached_z[0])
        self.dz = da * s * (1 - s)
        return self.dz

    def get_activation(self):
        return self.activation
