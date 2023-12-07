from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import sklearn.model_selection as model_selection
from typing import Union, Dict, List


ModelsUnion = Union[RandomForestClassifier, GradientBoostingClassifier]


class Model:
    models: Dict[str, ModelsUnion] = {}

    def __init__(self):
        pass

    @staticmethod
    def train_model(model_type : str,
                    params: dict,
                    features: List[List[int]],
                    labels: List[int]):

        if model_type == "random forest":
            model = RandomForestClassifier(**params)

        elif model_type == "gradient boosting":
            model = GradientBoostingClassifier(**params)

        else:
            return "you fool!"

        x_train, x_test, y_train, y_test = model_selection.train_test_split(
            features, labels, test_size=0.2, random_state=42,
        )

        model.fit(x_train, y_train)
        Model.models[model_type] = model
        return model_type

    @staticmethod
    def predict(model_type, features):
        if model_type not in Model.models:
            return "model are not available"

        model = Model.models[model_type]
        prediction = model.predict(features)
        return prediction.tolist()

    @staticmethod
    def delete_model(model_id):
        if Model.models[model_id]:
            del Model.models[model_id]
        else:
            return "Error"
