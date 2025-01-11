import typing as t
# import numpy as np
import torch
# import torch.nn as nn
# import torch.optim as optim
from .utils import WeakClassifier


class AdaBoostClassifier:
    def __init__(self, input_dim: int, num_learners: int = 10) -> None:
        self.sample_weights = None
        # create 10 learners, dont change.
        self.learners = [
            WeakClassifier(input_dim=input_dim) for _ in range(num_learners)
        ]
        self.alphas = []

    def fit(self, X_train, y_train, num_epochs: int = 500, learning_rate: float = 0.01):
        """Implement your code here"""
        epsilon = 1e-10
        n_samples, n_features = X_train.shape
        self.sample_weights = torch.ones(n_samples, 1) / n_samples  # Initialize weights
        losses_of_models = []
        # Convert data to PyTorch tensors with weights
        X_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        # weights_tensor = torch.tensor(self.sample_weights, dtype=torch.float32)

        for model in self.learners:

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            # Train weak learner
            for epoch in range(num_epochs):
                output = model(X_tensor)
                outputs = torch.sigmoid(output)
                first = y_tensor * torch.log(outputs + epsilon)
                second = (1 - y_tensor) * torch.log(1 - outputs + epsilon)
                loss = -torch.mean(
                    self.sample_weights * (first + second)
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Compute predictions and weighted error
            with torch.no_grad():
                outputs = torch.sigmoid(output)
                predictions = torch.round(outputs)
                incorrect = (predictions != y_tensor)
                error = torch.sum(self.sample_weights * incorrect) / torch.sum(self.sample_weights)

                # Compute alpha (learner weight)
            alpha = 0.5 * torch.log((1 - error + epsilon) / (error + epsilon))
            self.alphas.append(alpha.item())

            # Update sample weights
            self.sample_weights *= torch.exp(-alpha * y_tensor * (2 * predictions - 1))
            self.sample_weights /= torch.sum(self.sample_weights)

            # Track loss for the model
            losses_of_models.append(loss.item())
        return losses_of_models

    def predict_learners(self, X) -> t.Union[t.Sequence[int], t.Sequence[float]]:
        """Implement your code here"""

        probabilities = []
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        weight_sum = torch.zeros((X_tensor.shape[0], 1), dtype=torch.float32)

        # 對每個弱分類器進行加權預測
        for alpha, learner in zip(self.alphas, self.learners):
            predictions = torch.sigmoid(learner(X_tensor))

            probability = predictions.squeeze().tolist()
            probabilities.append(probability)

            predictions = 2 * torch.round(predictions) - 1
            weight_sum += alpha * predictions

        # 最終預測
        y_pred_classes = (weight_sum >= 0).squeeze().int().tolist()
        y_pred_probs = probabilities
        return y_pred_classes, y_pred_probs
        # raise NotImplementedError

    def compute_feature_importance(self) -> t.Sequence[float]:
        """Implement your code here"""
        n_features = self.learners[0].fc1.in_features
        feature_importance = torch.zeros(n_features, dtype=torch.float32)

        for alpha, learner in zip(self.alphas, self.learners):
            with torch.no_grad():
                weights = learner.fc1.weight.abs().sum(dim=0)
                feature_importance += alpha * weights
        feature_importance /= feature_importance.sum()
        return feature_importance.tolist()
        # raise NotImplementedError
