import typing as t
import numpy as np
import torch
# import torch.nn as nn
# import torch.optim as optim
from .utils import WeakClassifier


class BaggingClassifier:
    def __init__(self, input_dim: int) -> None:
        # create 10 learners, dont change.
        self.learners = [
            WeakClassifier(input_dim=input_dim) for _ in range(10)
        ]

    def fit(self, X_train, y_train, num_epochs: int, learning_rate: float):
        """Implement your code here"""
        losses_of_models = []
        n_samples = X_train.shape[0]
        # indices = n_samples

        for model in self.learners:
            sample_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_samples = X_train.iloc[sample_indices]
            y_samples = y_train[sample_indices]
            epsilon = 1e-10

            X_tensor = torch.tensor(X_samples.values, dtype=torch.float32)
            y_tensor = torch.tensor(y_samples, dtype=torch.float32).unsqueeze(1)

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            for epoch in range(num_epochs):
                outputs = torch.sigmoid(model(X_tensor))
                loss = -torch.mean(
                    y_tensor * torch.log(outputs + epsilon) + (1 - y_tensor) * torch.log(1 - outputs + epsilon)
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses_of_models.append(loss.item())
        return losses_of_models

    def predict_learners(self, X) -> t.Union[t.Sequence[int], t.Sequence[float]]:
        """Implement your code here"""
        predictions = []
        probabilities = []
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        for model in self.learners:
            with torch.no_grad():
                m = model(X_tensor)
                prediction = torch.sigmoid(m).squeeze()

                probability = prediction.tolist()
                probabilities.append(probability)

                prediction = prediction.numpy()
                predictions.append(prediction)
        avg = np.mean(predictions, axis=0)
        y_pred_classes = (avg >= 0.5).astype(int).tolist()
        y_pred_probs = probabilities
        return y_pred_classes, y_pred_probs
        # raise NotImplementedError

    def compute_feature_importance(self) -> t.Sequence[float]:
        """Implement your code here"""

        n_features = self.learners[0].fc1.in_features
        feature_importance = torch.zeros(n_features, dtype=torch.float32)

        for learner in self.learners:
            with torch.no_grad():
                weights = learner.fc1.weight.abs().sum(dim=0)
                feature_importance += weights

        feature_importance = feature_importance / feature_importance.sum()
        return feature_importance.tolist()

        # raise NotImplementedError
