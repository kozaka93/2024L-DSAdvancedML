class Optimizer:
    def initialize(self, dims):
        raise NotImplementedError

    def update(self, weights, gradient):
        raise NotImplementedError