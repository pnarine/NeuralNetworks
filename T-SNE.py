import numpy as np
class Myt_SNE:
    def __init__(self, perplexity, n_iters=1000, learn_rate=0.1):
        self.perplexity = perplexity
        self.n_iters = n_iters
        self.learn_rate = learn_rate

    def fit(self, X):
        n = len(X)
        y = np.random.normal(loc=0, scale=0.01, size=(n, 2))
        momentum = np.zeros(self.n_iters)
        Y_upd = []  # for gradient descent iterations we must have first 2
        Y_upd.append(y)
        Y_upd.append(y)

        for t in range(self.n_iters):
            if t < 250:
                momentum[t] = 0.5
            else:
                momentum[t] = 0.8

            y1 = Y_upd[t - 1] + self.learn_rate * self.grad(Y_upd[t - 1], self.prob_p(X, sigma), self.joint_prob_q(y)) + momentum[t]*(
                        Y_upd[t - 1] - Y_upd[t - 2])
            Y_upd.append(y1)

        return Y_upd

    def joint_prob_q(self, y):
        n = len(y)
        q1 = np.zeros((n, n))
        sum_n = 0
        for i in range(n):
            for j in range(n):
                q1[i, j] = 1 / (1 + (np.linalg.norm(y[i] - y[j])) ** 2)
                if (i != j):
                    sum_n += q1[i, j]
        q = q1 / sum_n
        return q

    def prob_p(self, X, sigma):  # pj|i
        n = len(X)
        p = np.zeros((n, n))
        cond_prob1 = np.zeros((n, n))
        cond_prob = []
        sum_i = 0
        for j in range(n):
            sum_i = 0
            for i in range(n):
                cond_prob1[j, i] = np.exp(-((np.linalg.norm(X[i] - X[j]) ** 2) / (2 * (sigma[i] ** 2))))
                if i != j:
                    sum_i += cond_prob1[j, i]
                cond_prob.append(cond_prob1[i] / sum_i)
        p = (cond_prob + cond_prob.T) / (2 * n)

        return p

    def grad(self, y, p, q):
        n = len(y)
        df_dy = np.zeros(n)
        sum_j = 0
        for i in range(n):
            for j in range(n):
                sum_j += (p[i, j] - q[i, j]) * (y[i] - y[j]) * (1 / (1 + (np.linalg.norm(y[i] - y[j])) ** 2))
                df_dy[i] = 4 * sum_j

                return df_dy

    def perpl(self, X):  # Perp(Pi)

    def sigma(self, X):

    def binary_search(self, a, v):  # find v in array a with precision eps
        eps = 1e-05
        if len(a) == 1:
            return int(a[0] == v) - 1
        i = len(a) // 2
        if abs(v - a[i]) < eps:
            return i
        elif v < a[i]:
            return binary_search(self, a[:i], v)
        else:
            j = binary_search(self, a[i:], v)
            if j == -1:
                return j
            return i + j


