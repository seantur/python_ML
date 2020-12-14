import numpy as np


class AdalineGD:

    def __init__(self, alpha=0.01, n_iter=50, seed=0):
        self.alpha = alpha
        self.n_iter = n_iter
        self.seed = seed

    def fit(self, X, y):
        """Fit training data.

        X: [n_examples, n_features], training vectors
        Y: [n_examples], target values
        """

        np.random.seed(self.seed)
        self.w_ = np.random.normal(scale=0.01, size=1 + X.shape[1])

        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.alpha * X.T.dot(errors)
            self.w_[0] += self.alpha * errors.sum()
            self.cost_.append((errors**2).sum() / 2)

    def net_input(self, X):
        """Calculate net input."""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation."""
        return X

    def predict(self, X):
        """Return class label after unit step."""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)


class AdalineSGD(AdalineGD):

    def __init__(self, alpha=0.01, n_iter=50, seed=0, shuffle=True):
        self.alpha = alpha
        self.n_iter = n_iter
        self.seed = seed
        self.shuffle = True
        self.w_initialized = False

    def fit(self, X, y):
        """Fit training data.

        X: [n_examples, n_features], training vectors
        Y: [n_examples], target values
        """

        self._initialize_weights(X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)

            cost = []

            for xi, y_hat in zip(X, y):
                cost.append(self._update_weights(xi, y_hat))

            self.cost_.append(np.mean(cost))

    def partial_fit(self, X, y):
        """Fit trianing data without reinitializing weights."""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, y_hat in zip(X, y):
                self._update_weights(xi, y_hat)
        else:
            self._update_weights(X, y)

    def _initialize_weights(self, m):
        np.random.seed(self.seed)
        self.w_ = np.random.normal(scale=0.01, size=1+m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        output = self.activation(self.net_input(xi))
        error = target - output
        self.w_[1:] += self.alpha * xi.dot(error)
        self.w_[0] + self.alpha * error
        return 0.5 * error**2

    def _shuffle(self, X, y):
        """Shuffle training data."""
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def net_input(self, X):
        """Calculate net input."""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation."""
        return X

    def predict(self, X):
        """Return class label after unit step."""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)

