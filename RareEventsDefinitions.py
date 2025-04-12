import numpy as np
from scipy.spatial.distance import cdist
from conjugate_gradient import conjugate_gradient
# from scipy.sparse.linalg import cg

def rbf_kernel(X1, X2, sigma):
    sq_dists = cdist(X1, X2, 'sqeuclidean')
    return np.exp(-sq_dists / (2 * sigma ** 2))


def loglikelihood(α, y, K, w, λ):
    LL = 0
    η = K @ α
    reg = (λ / 2) * α.T @ K @ α
    
    for i in range(len(y)):
        num = np.exp(y[i] * η[i])
        denom = 1 + np.exp(η[i])
        LL += w[i] * np.log(num / denom)
        
    # Add regularization term
    LL -= reg 
    return LL 

def deviance(α, y, K, w, λ):
    return -2 * loglikelihood(α, y, K, w, λ)

def weight_vector(w0, w1, y):
    return w1 * y + w0 * (1 - y)

# def RE_WKLR(X, y, w, σ, λ, max_iter=1000, tol=1e-6):


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


    
def calculate_weights(tau, resp):
    ybar = np.mean(resp)
    
    w0 = (1-tau)/(1-ybar)
    w1 = tau/(ybar)
    return w0, w1

# def WKLR(K, y, w1, w0, lam, max_iter=100, epsilon=1e-6):

#     alpha = np.zeros(K.shape[1])
#     bias = np.zeros(K.shape[1])
#     c = 0
#     DEV = [np.inf]


#     while c < max_iter:
#         K_alpha = K @ alpha
#         p_hat = sigmoid(K_alpha)
#         v = p_hat * (1 - p_hat)
#         w = weight_vector(w0, w1, y)
#         z = K_alpha + (y - p_hat) / (v + 1e-12)
#         Q_diag = 1 / (v * w + 1e-12)
#         xi = 0.5 / Q_diag * ((1 + w1) * p_hat - w1)

#         D = np.diag(v * w)

#         A = K.T @ D @ K + lam * K
#         b_alpha = K.T @ D @ z           # for alpha
#         b_bias = K.T @ D @ xi           # for bias

#         alpha_new = conjugate_gradient(A, b_alpha, alpha)
#         B_alpha = conjugate_gradient(A, b_bias, bias)

#         # Deviance check
#         pred_loss = -np.sum(w * (y * np.log(p_hat + 1e-12) + (1 - y) * np.log(1 - p_hat + 1e-12)))
#         DEV.append(pred_loss)

#         if abs(DEV[-2] - DEV[-1]) / (DEV[-1] + 1e-12) < epsilon:
#             break

#         alpha = alpha_new
#         c += 1

#     alpha_unbiased = alpha - B_alpha
#     p_opt = sigmoid(K @ alpha_unbiased)

#     return alpha, B_alpha, alpha_unbiased, p_opt




def WKLR(K, y, w1, w0, lambda_, epsilon_1=2.5, max_iter=30):
    
    alpha = alpha = np.zeros(K.shape[1])
    devs = []
    c = 0

    while c < max_iter:
        eta = K @ alpha
        p_hat = sigmoid(eta)
        v = p_hat * (1 - p_hat)
        w = w1 * y + w0 * (1 - y)

        z = eta + (y - p_hat) / (v + 1e-12)
        Q_diag = 1 / (v * w + 1e-12)
        xi = 0.5 * ((1 + w1) * p_hat - w1) / Q_diag

        D = np.diag(v * w)

        A = K.T @ D @ K + lambda_ * K
        b_alpha = K.T @ D @ z
        b_bias = K.T @ D @ xi

        alpha_new, _ = conjugate_gradient(A, b_alpha, x0=alpha)
        bias_alpha, _ = conjugate_gradient(A, b_bias, x0=np.zeros_like(alpha))

        dev_curr = deviance(alpha_new, y, K, w, lambda_)
        devs.append(dev_curr)

        if c > 0:
            rel_diff = np.abs(devs[c-1] - devs[c]) / (devs[c] + 1e-12)
            if rel_diff < epsilon_1:
                break

        alpha = alpha_new
        c += 1

    alpha_tilde = alpha - bias_alpha
    p_tilde = sigmoid(K @ alpha_tilde)

    return alpha, bias_alpha, alpha_tilde, p_tilde

def model_func(X_train, y_train, X_test, sigma, lambda_, tau, max_iter=1000):
    # Calculate the kernel matrix
    K_train = rbf_kernel(X_train, X_train, sigma)
    K_test = rbf_kernel(X_test, X_train, sigma)

    # Calculate weights
    w0, w1 = calculate_weights(tau, y_train)

    # Fit the model
    alpha, bias_alpha, alpha_tilde, p_tilde = WKLR(K_train, y_train, w1, w0, lambda_, max_iter=max_iter)

    # Make predictions on the test set
    pred_probs = sigmoid(K_test @ alpha_tilde)
    
    return pred_probs

