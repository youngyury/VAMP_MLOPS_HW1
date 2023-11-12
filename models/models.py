from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from typing import List
import sklearn.model_selection as model_selection
import sklearn.metrics as metrics


def predict_model(params_rf: dict,
                  params_gb: dict,
                  features: List[List[int]],
                  labels: List[int]):
    """
    Get accuracy from random forest and gradient boosting

    :param params_rf: hyperparameters for random forest
    :param params_gb: hyperparameters for gradient boosting
    :param features: list of feature vectors
    :param labels: list of labels
    :return: accuracy for random forest and gradient boosting
    """
    accuracy_rf = _predict_random_forest(params_rf, features, labels)
    accuracy_gb = _predict_gradboosting(params_gb, features, labels)
    return accuracy_rf, accuracy_gb


def _predict_random_forest(params: dict,
                           features: List[List[int]],
                           labels: List[int]):
    """
    Train random forest model and get prediction

    :param params: hyperparameters for random forest
    :param features: list of feature vectors
    :param labels: list of labels
    :return: accuracy for random forest
    """
    model = RandomForestClassifier(**params)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        features, labels, test_size=0.2, random_state=42,
    )

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    return accuracy


def _predict_gradboosting(params: dict,
                          features: List[List[int]],
                          labels: List[int]):
    """
    Train gradient boosting model and get prediction

    :param params: hyperparameters for gradient boosting
    :param features: list of feature vectors
    :param labels: list of labels
    :return: accuracy for gradient boosting
    """
    model = GradientBoostingClassifier(**params)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        features, labels, test_size=0.2, random_state=42,
    )

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    return accuracy
