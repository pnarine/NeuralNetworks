import numpy as np
from sklearn.cluster import KMeans


class MySpectralClustering:
    def __init__(self, k, method='normalized'):
        self.k = k
        self.method = method

    def Laplacians(self, X):
        n = len(X)
        norms = np.zeros((n, n))
        W = np.zeros((n, n))
        D = np.zeros((n, n))
        D12 = np.zeros((n, n))  # D^(-1/2)
        I = np.identity(n)
        # define similarity matrix with formula exp(-norm(x[i]-x[j])^2 / 2sig^2) where sigma = median(norm(x[i]-x[j]))
        for i in range(n):
            for j in range(n):
                norms[i, j] = np.linalg.norm(X[i] - X[j])
        sigma = np.median(norms)
        for i in range(n):
            for j in range(n):
                W[i, j] = np.exp(-pow(norms[i, j], 2) / 2 * (sigma ** 2))
            D[i, i] = np.sum(W[i, :])
            D12[i, i] = 1 / np.sqrt(D[i, i])

        # initialize laplacians
        if self.method == 'unnormalized':
            return D - W
        if self.method == 'normalized':
            return I - np.dot(np.dot(D12, W), D12)

    def fit(self, X):
        L = self.Laplacians(X)
        eiVal, eiVecs = np.linalg.eig(L)
        eiVecs = eiVecs[:, np.argsort(eiVal)]
        kVecs = eiVecs[:, 1:self.k]
        kmeans = KMeans(n_clusters=self.k)
        kmeans.fit(kVecs)
        self.clusters = kmeans.labels_

    def predict(self, X):
        return self.clusters