from pydantic import BaseModel, field_validator
from typing import List, Union, Optional


class TrainingData(BaseModel):
    """
    Schema for training data

    :param features: list of feature vectors
    :param labels: list of labels
    """
    features: List[List[float]]
    labels: List[str]


class RFParams(BaseModel):
    """
    Schema for random forest hyperparameters
    :param n_estimators: numbers of trees
    :param max_depth: max depth of trees
    """
    n_estimators: Optional[int] = 100
    max_depth: Optional[Union[int, None]] = None

    @field_validator('n_estimators')
    def n_estimators_positive(cls, n_estimators):
        if n_estimators < 0:
            raise ValueError('n_estimators must be positive')
        return n_estimators

    @field_validator('max_depth')
    def max_depth_positive(cls, max_depth):
        if max_depth < 0:
            raise ValueError('max_depth must be positive')
        return max_depth


class GBParams(BaseModel):
    """
    Schema for gradient boosting hyperparameters

    :param n_estimators: numbers of trees
    :param max_depth: max depth of trees
    :param learning_rate: speed of learning
    """
    n_estimators: Optional[int] = 100
    max_depth: Optional[Union[int, None]] = None
    learning_rate: Optional[float] = None

    @field_validator('n_estimators')
    def n_estimators_positive(cls, n_estimators):
        if n_estimators < 0:
            raise ValueError('n_estimators must be positive')
        return n_estimators

    @field_validator('max_depth')
    def max_depth_positive(cls, max_depth):
        if max_depth < 0:
            raise ValueError('max_depth must be positive')
        return max_depth

    @field_validator('learning_rate')
    def learning_rate_positive(cls, learning_rate):
        if learning_rate < 0:
            raise ValueError('learning_rate must be positive')
        return learning_rate


class ModelType(BaseModel):
    model_type: Optional[str]


class PredictionData(BaseModel):
    features: List[List[float]]