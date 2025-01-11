import pandas as pd
from loguru import logger
import random
import numpy as np
import torch
from src import AdaBoostClassifier, BaggingClassifier, DecisionTree
from src.utils import preprocess, plot_learners_roc
import matplotlib.pyplot as plt


def main():
    """
    Note:
    1) Part of line should not be modified.
    2) You should implement the algorithm by yourself.
    3) You can change the I/O data type as you need.
    4) You can change the hyperparameters as you want.
    5) You can add/modify/remove args in the function, but you need to fit the requirements.
    6) When plot the feature importance, the tick labels of one of the axis should be feature names.
    """
    random.seed(777)  # DON'T CHANGE THIS LINE
    torch.manual_seed(777)  # DON'T CHANGE THIS LINE
    train_df = pd.read_csv('./train.csv')
    test_df = pd.read_csv('./test.csv')
    category = train_df.columns.tolist()
    category.pop()
    X_train = train_df.drop(['target'], axis=1)
    y_train = train_df['target'].to_numpy()  # (n_samples, )

    X_test = test_df.drop(['target'], axis=1)
    y_test = test_df['target'].to_numpy()
    # (TODO): Implement you preprocessing function.
    X_train, x_train_feature_names = preprocess(X_train)
    X_test, x_test_feature_names = preprocess(X_test)

    """
    (TODO): Implement your ensemble methods.
    1. You can modify the hyperparameters as you need.
    2. You must print out logs (e.g., accuracy) with loguru.
    """
    # AdaBoost
    clf_adaboost = AdaBoostClassifier(
        input_dim=X_train.shape[1],
    )
    _ = clf_adaboost.fit(
        X_train,
        y_train,
        num_epochs=40,
        learning_rate=0.009,
    )
    y_pred_classes, y_pred_probs = clf_adaboost.predict_learners(X_test)
    accuracy_ = np.mean(y_pred_classes == y_test)
    logger.info(f'AdaBoost - Accuracy: {accuracy_:.4f}')
    plot_learners_roc(
        y_preds=y_pred_probs,
        y_trues=y_test,
        fpath="D:\\user\\Desktop\\for_school\\machine_learning\\hw3\\adaboost.jpg",
    )
    feature_importance = clf_adaboost.compute_feature_importance()
    # # (TODO) Draw the feature importance
    draw_score, draw_feature = compute_average_feature_importance(feature_importance, x_train_feature_names, category)
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(draw_score)), draw_score)
    plt.xticks(range(len(draw_score)), draw_feature, rotation=45)
    # plt.xticks(range(len(draw_score)), draw_feature)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('./adaboost_feature_importance.png')
    # 顯示圖形
    plt.show()

    # Bagging
    clf_bagging = BaggingClassifier(
        input_dim=X_train.shape[1],
    )
    _ = clf_bagging.fit(
        X_train,
        y_train,
        num_epochs=40,
        learning_rate=0.009,
    )
    y_pred_classes, y_pred_probs = clf_bagging.predict_learners(X_test)
    accuracy_ = np.mean(y_pred_classes == y_test)
    logger.info(f'Bagging - Accuracy: {accuracy_:.4f}')
    plot_learners_roc(
        y_preds=y_pred_probs,
        y_trues=y_test,
        fpath="D:\\user\\Desktop\\for_school\\machine_learning\\hw3\\bagging.jpg",
    )
    feature_importance = clf_bagging.compute_feature_importance()
    draw_score, draw_feature = compute_average_feature_importance(feature_importance, x_train_feature_names, category)
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(draw_score)), draw_score)
    plt.xticks(range(len(draw_score)), draw_feature, rotation=45)
    # plt.xticks(range(len(draw_score)), draw_feature)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('./bagging_feature_importance.png')
    # 顯示圖形
    plt.show()

    # Decision Tree
    clf_tree = DecisionTree(
        max_depth=7,
    )
    clf_tree.fit(X_train, y_train)
    y_pred_classes = clf_tree.predict(X_test)
    accuracy_ = np.mean(y_pred_classes == y_test)
    logger.info(f'DecisionTree - Accuracy: {accuracy_:.4f}')
    feature_importance = clf_tree.compute_feature_importance()
    draw_score, draw_feature = compute_average_feature_importance(feature_importance, x_train_feature_names, category)
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(draw_score)), draw_score)
    plt.xticks(range(len(draw_score)), draw_feature, rotation=45)
    # plt.xticks(range(len(draw_score)), draw_feature)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('./tree_feature_importance.png')
    # 顯示圖形
    plt.show()

    data = [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1]
    values, counts = np.unique(data, return_counts=True)
    probs = counts / counts.sum()
    gini = 1 - np.sum(probs ** 2)
    entropy = -np.sum(probs * np.log2(probs))
    print(f"Gini Index: {gini:.4f}")
    print(f"Entropy: {entropy:.4f}")


def compute_average_feature_importance(feature_importance, feature_names, category):
    importance = {}
    for feature in feature_names:
        if not any(feature.startswith(cat) for cat in category):
            importance[feature] = feature_importance[feature_names.index(feature)]
    for cat_feature in category:
        cat_related_features = [name for name in feature_names if name.startswith(cat_feature)]
        avg_importance = sum(
            feature_importance[feature_names.index(name)] for name in cat_related_features
        ) / len(cat_related_features)
        importance[cat_feature] = avg_importance
    merged_feature_names = list(importance.keys())
    merged_importance = list(importance.values())
    return merged_importance, merged_feature_names


if __name__ == '__main__':
    main()
