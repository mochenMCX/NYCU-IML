"""
You dont have to follow the stucture of the sample code.
However, you should checkout if your class/function meet the requirements.
"""
import numpy as np
import pandas as pd
import typing as t


class DecisionTree:
    def __init__(self, max_depth=1):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)
        self.tree["n_features"] = X.shape[1]

    def _grow_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        num_classes = len(np.unique(y))

        if depth >= self.max_depth or num_classes == 1 or num_samples <= 1:
            leaf_value = np.bincount(y).argmax()
            return {"leaf": True, "value": leaf_value, "num_samples": num_samples, "impurity": gini(y)}

        feature_index, threshold = find_best_split(X, y)

        if feature_index is None:
            leaf_value = np.bincount(y).argmax()
            return {"leaf": True, "value": leaf_value, "num_samples": num_samples, "impurity": gini(y)}
        left_indeices, right_indices = split_dataset(X, y, feature_index, threshold)
        lsub = self._grow_tree(X.iloc[left_indeices], y[left_indeices], depth + 1)
        rsub = self._grow_tree(X.iloc[right_indices], y[right_indices], depth + 1)

        return {
            "leaf": False,
            "feature": feature_index,
            "threshold": threshold,
            "left": lsub,
            "right": rsub,
            "num_samples": num_samples,
            "impurity": gini(y)
        }
        # raise NotImplementedError

    def predict(self, X):
        X = X.values
        prediction = [self._predict_tree(i, self.tree) for i in X]
        return np.array(prediction)
        # raise NotImplementedError

    def _predict_tree(self, x, tree_node):
        if tree_node["leaf"]:
            return tree_node["value"]

        feature = tree_node["feature"]
        threshold = tree_node["threshold"]

        if x[feature] <= threshold:
            return self._predict_tree(x, tree_node["left"])
        else:
            return self._predict_tree(x, tree_node["right"])
        # raise NotImplementedError

    def compute_feature_importance(self) -> t.Sequence[float]:
        n_features = self.tree["n_features"]
        feature_importance = np.zeros(n_features, dtype=np.float32)

        def _traverse_and_compute(tree_node, parent_samples, parent_impurity):
            if tree_node["leaf"]:
                return

            feature = tree_node["feature"]
            left = tree_node["left"]
            right = tree_node["right"]

            left_samples = left["num_samples"]
            right_samples = right["num_samples"]

            le = (left_samples / parent_samples)
            r = (right_samples / parent_samples)
            weighted_impurity = (le * left["impurity"] + r * right["impurity"])

            impurity_reduction = parent_impurity - weighted_impurity
            feature_importance[feature] += impurity_reduction * parent_samples

            _traverse_and_compute(left, left_samples, left["impurity"])
            _traverse_and_compute(right, right_samples, right["impurity"])

        _traverse_and_compute(self.tree, self.tree["num_samples"], self.tree["impurity"])

        feature_importance /= feature_importance.sum()

        return feature_importance.tolist()


# Split dataset based on a feature and threshold
def split_dataset(X, y, feature_index, threshold):
    if isinstance(X, pd.DataFrame):
        left_indices = np.where(X.iloc[:, feature_index] <= threshold)[0]
        right_indices = np.where(X.iloc[:, feature_index] > threshold)[0]
    else:
        left_indices = np.where(X[:, feature_index] <= threshold)[0]
        right_indices = np.where(X[:, feature_index] > threshold)[0]
    return left_indices, right_indices
    # raise NotImplementedError


# Find the best split for the dataset
def find_best_split(X, y):
    num_samples, num_features = X.shape
    best_feature, best_threshold = None, None
    best_gini = float("inf")

    for feature_index in range(num_features):
        if isinstance(X, pd.DataFrame):
            thresholds = np.unique(X.iloc[:, feature_index])
        else:
            thresholds = np.unique(X[:, feature_index])

        for threshold in thresholds:
            left_indices, right_indices = split_dataset(X, y, feature_index, threshold)

            if len(left_indices) == 0 or len(right_indices) == 0:
                continue

            left_gini = gini(y[left_indices])
            right_gini = gini(y[right_indices])

            n_left, n_right = len(left_indices), len(right_indices)
            weighted_gini = (n_left / num_samples) * left_gini + (n_right / num_samples) * right_gini

            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feature = feature_index
                best_threshold = threshold

    return best_feature, best_threshold

    # raise NotImplementedError


def entropy(y):
    unique_counts = np.bincount(y)
    probabilities = unique_counts / len(y)
    probabilities = probabilities[probabilities > 0]
    return -np.sum(probabilities * np.log2(probabilities))
    # raise NotImplementedError


def gini(y):
    unique_counts = np.bincount(y)
    probabilities = unique_counts / len(y)
    return 1 - np.sum(probabilities ** 2)
