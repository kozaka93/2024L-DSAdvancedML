import numpy as np


class MaxIterationCondition:
    def __init__(self, max_iterations=10):
        self.max_iterations = max_iterations
        self.iteration = 0

    def __call__(self, **kwargs):
        self.iteration += 1
        if self.iteration > self.max_iterations:
            return True
        return False


class NoLogLikImprovementCondition:
    def __init__(self, patience=3):
        self.epoch = -2
        self.last_improvement_epoch = 0
        self.best_score = 1e10
        self.patience = patience
        self.best_model = None

    def __call__(self, model=None, x=None, y=None, **kwargs):
        self.best_model = self.best_model or model
        self.epoch += 1
        if self.epoch < 0:
            return False
        prediction = np.clip(model.predict_probs(x), 1e-10, 1 - 1e-10)
        loglik = - np.mean(y * np.log(prediction) + (1 - y) * np.log(1 - prediction))
        if loglik >= self.best_score and self.epoch - self.last_improvement_epoch >= self.patience:
            return True
        elif loglik < self.best_score:
            self.best_score = loglik
            self.last_improvement_epoch = self.epoch
            self.best_model = model
        return False


class NoLogLikOrMaxIterCondition(NoLogLikImprovementCondition):
    def __init__(self, max_iterations=10, patience=3):
        super().__init__(patience)
        self.max_iterations = max_iterations

    def __call__(self, model=None, x=None, y=None, **kwargs):
        return self.epoch > self.max_iterations or super().__call__(model=model, x=x, y=y, **kwargs)
