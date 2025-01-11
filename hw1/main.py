import numpy as np
import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt


class LinearRegressionBase:
    def __init__(self):
        self.weights = None
        self.intercept = None

    def fit(self):
        return
        # raise NotImplementedError

    def predict(self):
        return
        # raise NotImplementedError


class LinearRegressionCloseform(LinearRegressionBase):
    def fit(self, X, y):
        m = X.shape[0]
        ones = np.ones((m, 1))
        X_extend = np.append(ones, X, axis=1)
        result = np.linalg.inv((X_extend.T @ X_extend)) @ X_extend.T @ y
        self.intercept = result[0]
        self.weights = result[1:]
        return
        # raise NotImplementedError

    def predict(self, X):
        return X @ self.weights.reshape(-1, 1) + self.intercept
        # raise NotImplementedError


class LinearRegressionGradientdescent(LinearRegressionBase):
    def fit(self, X, y, learning_rate: float = 0.01, epochs: int = 1000):
        m = X.shape[0]
        X = np.hstack((np.ones((m, 1)), X))
        n = X.shape[1]
        theta = np.zeros((n, 1))
        y = y.reshape(m, 1)
        losses = []
        for i in range(epochs):
            error = (X @ theta) - y
            MSE = (error.T @ error) * 1 / m
            losses.append(MSE[0][0])
            theta = theta - (learning_rate * 2 / m) * (X.T @ error)
        error = (X @ theta) - y
        MSE = MSE = (error.T @ error) * 1 / m
        losses.append(MSE[0][0])
        self.intercept = theta.T[0][0]
        self.weights = theta.T[0][1:]
        # self.intercept = np.float64(self.intercept[0])
        return losses
        # raise NotImplementedError

    def predict(self, X):
        return X @ self.weights.reshape(-1, 1) + self.intercept
        # raise NotImplementedError

    def plot_learning_curve(self, losses):
        plt.plot(losses, label="Train MSE loss")
        plt.legend()
        plt.title("Training loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.show()
        # raise NotImplementedError


def compute_mse(prediction, ground_truth):
    m = len(ground_truth)
    result = 0
    result = np.sum((prediction.T - ground_truth) ** 2) / m
    return result
    # raise NotImplementedError


def main():
    # read data
    train_df = pd.read_csv('./train.csv')
    train_x = train_df.drop(["Performance Index"], axis=1).to_numpy()
    train_y = train_df["Performance Index"].to_numpy()

    LR_CF = LinearRegressionCloseform()
    LR_CF.fit(train_x, train_y)
    logger.info(f'{LR_CF.weights=}, {LR_CF.intercept=:.4f}')

    LR_GD = LinearRegressionGradientdescent()
    losses = LR_GD.fit(train_x, train_y, learning_rate=1e-4, epochs=1000000)
    LR_GD.plot_learning_curve(losses)
    logger.info(f'{LR_GD.weights=}, {LR_GD.intercept=:.4f}')

    test_df = pd.read_csv('./test.csv')
    test_x = test_df.drop(["Performance Index"], axis=1).to_numpy()
    test_y = test_df["Performance Index"].to_numpy()

    y_preds_cf = LR_CF.predict(test_x)
    y_preds_gd = LR_GD.predict(test_x)
    y_preds_diff = np.abs(y_preds_gd - y_preds_cf).mean()
    logger.info(f'Mean prediction difference: {y_preds_diff:.4f}')

    mse_cf = compute_mse(y_preds_cf, test_y)
    mse_gd = compute_mse(y_preds_gd, test_y)
    diff = (np.abs(mse_gd - mse_cf) / mse_cf) * 100
    logger.info(f'{mse_cf=:.4f}, {mse_gd=:.4f}. Difference: {diff:.3f}%')


if __name__ == '__main__':
    main()
