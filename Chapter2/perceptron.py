import numpy as np

class Perceptron:

    def __init__(self, alpha=0.1, n_iter=50, seed=0):
        self.alpha = alpha
        self.n_iter = n_iter
        self.seed = seed

    def fit(self, X, Y):
        """Fit training data.

        X: [n_examples, n_features], training vectors
        Y: [n_examples], target values
        """

        np.random.seed(self.seed)
        self.w_ = np.random.normal(scale=0.01, size=1 + X.shape[1])

        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, Y):
                y_hat = self.predict(xi)
                update = self.alpha * (target - y_hat)
                
                self.w_[1:] += update*xi
                self.w_[0] += update

                errors += (target != y_hat)

            self.errors_.append(errors)

    def net_input(self, X):
        """Calculate net input."""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step."""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
