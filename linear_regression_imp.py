import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class MyLinearRegression(object):

    def __init__(self, alpha=0.1):
        self.n = 0
        self.m = 0
        self.coeffs = None
        self.X = None
        self.y = None
        self.errors = None
        self.alpha = alpha


    @staticmethod
    def plot_data(X, y):
        plt.figure()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Linear Regression\nData Plot')
        plt.scatter(X, y, color = 'r', marker='x', s=20)

    def compute_cost(self):
        h_x = np.dot(self.X.T, self.coeffs)
        self.errors = h_x - self.y
        cost = np.sum(self.errors**2)/2*self.m
        return cost

    def gradient_decent(self):
        before_cost = np.inf
        after_cost = self.compute_cost()
        while(abs(before_cost-after_cost) > 0.001):
            #print(after_cost)
            self.update_coeffs()
            before_cost = after_cost
            after_cost = self.compute_cost()
            if after_cost > before_cost:
                print("Failed to converge. alpha might be too big")
                quit()
        return after_cost

    def feature_normalize(self):
        pass

    def normal_eqn(self):
        pass

    def update_coeffs(self):
        for i, _ in enumerate(self.coeffs):
            self.coeffs[i] = self.coeffs[i] - (self.alpha*np.sum(self.errors*self.X.T[:,i]))/self.m

    def fit(self, X, y):
        self.X = np.insert(X, 0, 1, 0)
        self.y = y
        self.m = X.shape[1]
        self.n = X.shape[0]
        if self.coeffs is None:
            self.coeffs = np.random.uniform(-1, 1, self.n + 1)

        after_cost = self.gradient_decent()

    def pred(self, X):
        X = np.insert(X, 0, 1, 0)
        return np.dot(X.T, self.coeffs)


### Consts

f1_dfile = 'ex1data1.txt'
f2_dfile = 'ex1data2.txt'

### Functions

def main():
    # changing main()
    # Check for update
    lr = MyLinearRegression(alpha=0.01)
    df = pd.read_csv(f1_dfile, header=None)
    data = np.asarray(df)
    X = data.T[:-1]
    y = data.T[-1:]
    # X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9,
    #               4, 5, 6, 7, 8, 7, 8, 9, 10]).reshape(2, 9)
    # y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(1, 9)
    lr.fit(X, y)
    y_pred = lr.pred(X)
    lr.plot_data(X, y)
    plt.plot(X.T, y_pred)
    plt.show()


    #MyLinearRegression.plot_data(X, y)

if __name__ == '__main__':
    main()