import numpy as np


class Parameter(np.ndarray):

    def __new__(cls, data=None, parameter=None):
        cls.dimension = data
        cls.parameter_init = parameter
        return cls.initialize_parameter(cls)

    def initialize_parameter(self):
        np.random.seed(1)
        if self.parameter_init == "xavier":
            parameter = np.random.randn(self.dimension[1], self.dimension[0]) * np.sqrt(1.0 / (self.dimension[0]+self.dimension[1]))
        elif self.parameter_init == "Zero":
            parameter = np.zeros([self.dimension[1], self.dimension[0]])
        else:
            parameter = np.random.randn(self.dimension[1], self.dimension[0]) / np.sqrt(self.dimension[0])
        # print("param ", parameter)
        # print('Param ',self.dimension[1], self.dimension[0])
        return parameter
