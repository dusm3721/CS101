import numpy as np
import matplotlib.pyplot as plt

class LinearRegression():
    def __init__(self, reg_coef = 0):
        self.reg_coef = reg_coef
        self.beta = np.array([])

    def train(self, X, y):
        N = len(X)
        p = len(X[0])
        all_ones = np.ones([N, 1])
        X = np.concatenate((X, all_ones), axis=1)
        XTX = np.matmul(np.transpose(X), X)
        XTy = np.matmul(np.transpose(X), y)
        self.beta = np.matmul(np.linalg.inv(XTX + self.reg_coef*np.identity(p+1)), XTy)

    def predict(self, X):
        N = len(X)
        p = len(X[0])
        all_ones = np.ones([N, 1])
        X = np.concatenate((X, all_ones), axis=1)
        yhat = np.matmul(X, self.beta)
        return yhat

    def Rsquared(self, X, y):
        N = len(X)
        yhat = self.predict(X)
        uhat = y - yhat
        SSE = uhat.var() * N
        SST = y.var() * N
        return 1 - SSE/SST
    
    def accuracy(self, X, y):
        N = len(X)
        yhat = self.predict(X)
        n_correct = 0
        for i in range(N):
            if y[i] * yhat[i] > 0:
                n_correct +=1
        return n_correct / N

X_data = np.array([1400,1600,1700,1875,1100,1550,2350,2450,1425,1700])        
X_data = X_data.reshape([-1,1])
y_data = np.array([245,312,279,308,199,219,405,324,319,255])

model = LinearRegression(0)
model.train(X_data, y_data)
model.Rsquared(X_data, y_data)
yhat_data = model.predict(X_data)

plt.plot(X_data, y_data, "o", X_data, yhat_data)
