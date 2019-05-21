from . import layers


class Input:

    def __new__(cls, x):
        cls.x = x
        cls.input = layers.Input(x)
        return cls.input.forward_features, cls.input.set_features, cls.input.backward_features, 'input'


class Linear:

    def __new__(cls, in_features, out_features, parameter=None, bias=True, name='linear'):
        cls.layer_name = name
        cls.in_features = in_features
        cls.out_features = out_features
        cls.linear = layers.Linear(in_features, out_features)
        return cls.linear.forward, cls.linear.get_parameters, cls.linear.get_linear, cls.layer_name, cls.linear.set_parameters


class ReLU:

    def __new__(cls):
        cls.relu = layers.ReLU()
        return cls.relu.forward, cls.relu.backward, cls.relu.get_activation, 'relu'


class Sigmoid:
    def __new__(cls):
        cls.sigmoid = layers.Sigmoid()
        return cls.sigmoid.forward, cls.sigmoid.backward, cls.sigmoid.get_activation, 'sigmoid'
