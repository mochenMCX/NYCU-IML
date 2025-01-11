import typing as t

import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self, learning_rate: float = 1e-4, num_iterations: int = 100):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.intercept = None

    def fit(
        self,
        inputs: npt.NDArray[float],
        targets: t.Sequence[int],
    ) -> None:
        """
        Implement your fitting function here.
        The weights and intercept should be kept in self.weights and self.intercept.
        """
        n = inputs.shape[1]
        m = inputs.shape[0]
        self.weights = np.zeros((n, 1))
        self.intercept = 0.0
        targets = targets[:, np.newaxis]
        for i in range(self.num_iterations):
            z = inputs @ self.weights + self.intercept
            predictions = self.sigmoid(z)
            weight_loss = 1 / m * np.sum(((predictions - targets) * inputs), axis=0)
            weight_loss = weight_loss.reshape(1, -1)
            weight_loss = weight_loss.reshape(-1, 1)
            intercept_loss = 1 / m * np.sum(predictions - targets)
            self.weights -= self.learning_rate * weight_loss
            self.intercept -= self.learning_rate * intercept_loss
        # raise NotImplementedError

    def predict(
        self,
        inputs: npt.NDArray[float],
    ) -> t.Tuple[t.Sequence[np.float64], t.Sequence[int]]:
        """
        Implement your prediction function here.
        The return should contains
        1. sample probabilty of being class_1
        2. sample predicted class
        """
        input = inputs @ self.weights + self.intercept
        z = self.sigmoid(input)
        cla = np.where(z > 0.5, 1, 0)
        return z, cla
        # raise NotImplementedError

    def sigmoid(self, x):
        """
        Implement the sigmoid function.
        """
        return 1.0 / (1.0 + np.exp(-x))
        # raise NotImplementedError


class FLD:
    """Implement FLD
    You can add arguments as you need,
    but don't modify those already exist variables.
    """
    def __init__(self):
        self.w = None
        self.m0 = None
        self.m1 = None
        self.sw = None
        self.sb = None
        self.slope = None
        self.intercept = None

    def fit(
        self,
        inputs: npt.NDArray[float],
        targets: t.Sequence[int],
    ) -> None:
        c0 = inputs[targets == 0]
        c1 = inputs[targets == 1]
        self.m0 = np.mean(c0, axis=0)
        self.m1 = np.mean(c1, axis=0)
        self.m0 = self.m0.reshape(1, -1)
        self.m1 = self.m1.reshape(1, -1)
        self.sw = np.array([[0.0, 0.0], [0.0, 0.0]])
        for i in c0:
            cal = i - self.m0
            self.sw += np.outer(cal, cal)
        for i in c1:
            cal = i - self.m1
            self.sw += np.outer(cal, cal)
        cal = self.m1 - self.m0
        self.sb = np.outer(cal, cal)
        cal = cal.reshape(-1, 1)
        self.w = np.linalg.inv(self.sw) @ cal
        # raise NotImplementedError

    def predict(
        self,
        inputs: npt.NDArray[float],
    ) -> t.Sequence[t.Union[int, bool]]:
        projections = inputs @ self.w
        threshold = ((self.m0 @ self.w) + (self.m1 @ self.w)) / 2
        answer = (projections >= threshold).astype(int)
        return answer
        # raise NotImplementedError

    def plot_projection(self, inputs: npt.NDArray[float],):
        self.slope = self.w[1] / self.w[0]
        self.intercept = (self.m1[0, 1] - self.slope * self.m1[0, 0]).item()
        y_preds = self.predict(inputs)
        plt.figure(figsize=(8, 8))

        for idx, point in enumerate(inputs):
            color = 'blue' if y_preds[idx] == 1 else 'red'
            plt.scatter(point[0], point[1], c=color, alpha=0.5, marker='o')

            x_proj = ((point[1] - self.intercept + point[0] / self.slope) / (self.slope + 1 / self.slope)).item()
            y_proj = (self.slope * x_proj + self.intercept).item()
            plt.scatter(x_proj, y_proj, c=color, marker='o')
            plt.plot([point[0], x_proj], [point[1], y_proj], color='blue', linestyle='-', linewidth=0.5)

        x = np.linspace(plt.xlim()[0], plt.xlim()[1], 400)
        y = self.slope * x + self.intercept
        plt.plot(x, y, color='black', linestyle='-')
        x_min = inputs[:, 0].min()
        x_max = inputs[:, 0].max()
        y_min = inputs[:, 1].min()
        y_max = inputs[:, 1].max()
        plt.xlim(x_min - (x_max - x_min) * 0.1, x_max + (x_max - x_min) * 0.1)
        plt.ylim(y_min - (y_max - y_min) * 0.1, y_max + (y_max - y_min) * 0.1)
        self.slope = self.slope[0]
        plt.gca().set_aspect('auto')
        plt.title(f'Projection Line: w ={self.slope:.2f}, b={self.intercept:.2f}')
        plt.show()

        # raise NotImplementedError


def compute_auc(y_trues, y_preds):
    return roc_auc_score(y_trues, y_preds)
    # raise NotImplementedError


def accuracy_score(y_trues, y_preds):
    right = 0
    for i in range(len(y_trues)):
        if y_trues[i] == y_preds[i]:
            right += 1
    return right / len(y_trues)
    # raise NotImplementedError


def main():
    # Read data
    train_df = pd.read_csv('./train.csv')
    test_df = pd.read_csv('./test.csv')

    # Part1: Logistic Regression
    x_train = train_df.drop(['target'], axis=1).to_numpy()  # (n_samples, n_features)
    y_train = train_df['target'].to_numpy()  # (n_samples, )

    x_test = test_df.drop(['target'], axis=1).to_numpy()
    y_test = test_df['target'].to_numpy()

    LR = LogisticRegression(
        learning_rate=1e-2,  # You can modify the parameters as you want
        num_iterations=10000,  # You can modify the parameters as you want
    )
    LR.fit(x_train, y_train)
    y_pred_probs, y_pred_classes = LR.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred_classes)
    auc_score = compute_auc(y_test, y_pred_probs)
    logger.info(f'LR: Weights: {LR.weights[:5]}, Intercep: {LR.intercept}')
    logger.info(f'LR: Accuracy={accuracy:.4f}, AUC={auc_score:.4f}')

    # Part2: FLD
    cols = ['10', '20']  # Dont modify
    x_train = train_df[cols].to_numpy()
    y_train = train_df['target'].to_numpy()
    x_test = test_df[cols].to_numpy()
    y_test = test_df['target'].to_numpy()

    FLD_ = FLD()
    """
    (TODO): Implement your code to
    1) Fit the FLD model
    2) Make prediction
    3) Compute the evaluation metrics

    Please also take care of the variables you used.
    """
    FLD_.fit(x_train, y_train)
    y_pred_classes = FLD_.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred_classes)

    logger.info(f'FLD: m0={FLD_.m0}, m1={FLD_.m1} of {cols=}')
    logger.info(f'FLD: \nSw=\n{FLD_.sw}')
    logger.info(f'FLD: \nSb=\n{FLD_.sb}')
    logger.info(f'FLD: \nw=\n{FLD_.w}')
    logger.info(f'FLD: Accuracy={accuracy:.4f}')

    """
    (TODO): Implement your code below to plot the projection
    """
    FLD_.plot_projection(x_train)
    FLD_.plot_projection(x_test)


if __name__ == '__main__':
    main()
