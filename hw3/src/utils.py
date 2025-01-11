import typing as t
# import torch
# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def preprocess(df: pd.DataFrame):
    """
    (TODO): Implement your preprocessing function.
    """
    # Define numeric and categorical features.
    numeric_features = [
        'person_age', 'person_income', 'person_emp_exp', 'loan_amnt',
        'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
        'credit_score'
    ]
    categorical_features = [
        'person_gender', 'person_education', 'person_home_ownership',
        'loan_intent', 'previous_loan_defaults_on_file'
    ]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder()

    # Combine both transformers into a column transformer.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Fit and transform the data using the preprocessor.
    processed_data = preprocessor.fit_transform(df)

    # Get the feature names for numeric and categorical features.
    numeric_feature_names = numeric_features
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)

    # Combine numeric and categorical feature names into a single list.
    all_feature_names = numeric_feature_names + list(cat_feature_names)

    # Create a DataFrame with processed data and feature names as columns.
    df_processed = pd.DataFrame(processed_data, columns=all_feature_names)

    return df_processed, all_feature_names


class WeakClassifier(nn.Module):
    """
    Use pyTorch to implement a 1 ~ 2 layers model.
    Here, for example:
        - Linear(input_dim, 1) is a single-layer model.
        - Linear(input_dim, k) -> Linear(k, 1) is a two-layer model.

    No non-linear activation allowed.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 10):
        super(WeakClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1)
        # self.fc2 = nn.Linear(input_dim, hidden_dim)
        # self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        if hasattr(self, 'fc3'):
            x = self.fc2(x)
            x = self.fc3(x)
        else:
            x = self.fc1(x)
        # x = torch.sigmoid(x)
        return x


def accuracy_score(y_trues, y_preds) -> float:
    acc = 0
    for i in range(y_trues.shape):
        if y_trues[i] == y_preds[i]:
            acc + 1
    return acc / y_trues.shape
    # return np.mean(y_pred_classes == y_test)
    # raise NotImplementedError


def entropy_loss(outputs, targets):
    # Ensure targets are floats
    targets = targets.float()
    # Use binary cross-entropy with logits
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(outputs, targets)
    return loss

    # raise NotImplementedError


def plot_learners_roc(
    y_preds: t.List[t.Sequence[float]],
    y_trues: t.Sequence[int],
    fpath='./tmp.png',
):
    plt.figure()
    for idx, preds in enumerate(y_preds):
        fpr, tpr, _ = roc_curve(y_trues, preds)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'Learner {idx+1} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Learners')
    plt.legend(loc='lower right')
    plt.savefig(fpath)
    plt.show()
    plt.close()
    # raise NotImplementedError
