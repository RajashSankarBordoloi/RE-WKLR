import numpy as np
from scipy.spatial.distance import cdist
from conjugate_gradient import conjugate_gradient

class WeightedKernelLogisticRegression:
    def __init__(self, sigma=1.0, lambda_=0.1, tau=0.5, max_iter=30, epsilon=2.5):
        self.sigma = sigma
        self.lambda_ = lambda_
        self.tau = tau
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.alpha_tilde = None
        self.X_train = None

    def _rbf_kernel(self, X1, X2):
        sq_dists = cdist(X1, X2, 'sqeuclidean')
        return np.exp(-sq_dists / (2 * self.sigma ** 2))

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _calculate_weights(self, y):
        ybar = np.mean(y)
        w0 = (1 - self.tau) / (1 - ybar)
        w1 = self.tau / ybar
        return w0, w1

    def _weight_vector(self, w0, w1, y):
        return w1 * y + w0 * (1 - y)

    def _loglikelihood(self, alpha, y, K, w):
        eta = K @ alpha
        reg = (self.lambda_ / 2) * alpha.T @ K @ alpha
        ll = 0
        for i in range(len(y)):
            num = np.exp(y[i] * eta[i])
            denom = 1 + np.exp(eta[i])
            ll += w[i] * np.log(num / denom)
        return ll - reg

    def _deviance(self, alpha, y, K, w):
        return -2 * self._loglikelihood(alpha, y, K, w)

    def _train_WKLR(self, K, y, w0, w1):
        alpha = np.zeros(K.shape[1])
        devs = []
        for c in range(self.max_iter):
            eta = K @ alpha
            p_hat = self._sigmoid(eta)
            v = p_hat * (1 - p_hat)
            w = self._weight_vector(w0, w1, y)

            z = eta + (y - p_hat) / (v + 1e-12)
            Q_diag = 1 / (v * w + 1e-12)
            xi = 0.5 * ((1 + w1) * p_hat - w1) / Q_diag

            D = np.diag(v * w)

            A = K.T @ D @ K + self.lambda_ * K
            b_alpha = K.T @ D @ z
            b_bias = K.T @ D @ xi

            alpha_new, _ = conjugate_gradient(A, b_alpha, x0=alpha)
            bias_alpha, _ = conjugate_gradient(A, b_bias, x0=np.zeros_like(alpha))

            dev_curr = self._deviance(alpha_new, y, K, w)
            devs.append(dev_curr)

            if c > 0:
                rel_diff = np.abs(devs[-2] - devs[-1]) / (devs[-1] + 1e-12)
                if rel_diff < self.epsilon:
                    break

            alpha = alpha_new

        alpha_tilde = alpha - bias_alpha
        return alpha, bias_alpha, alpha_tilde

    def fit(self, X_train, y_train):
        self.X_train = X_train
        K_train = self._rbf_kernel(X_train, X_train)
        w0, w1 = self._calculate_weights(y_train)
        _, _, self.alpha_tilde = self._train_WKLR(K_train, y_train, w0, w1)

    def predict_proba(self, X_test):
        if self.alpha_tilde is None:
            raise RuntimeError("Model not trained yet. Call fit() first.")
        K_test = self._rbf_kernel(X_test, self.X_train)
        return self._sigmoid(K_test @ self.alpha_tilde)

    def predict(self, X_test, threshold=0.5):
        return (self.predict_proba(X_test) >= threshold).astype(int)
