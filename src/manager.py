from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import sklearn.model_selection as model_selection
from typing import Union, Dict, List
import boto3
import joblib
from io import BytesIO
from .dvc_data import DVC


ModelsUnion = Union[RandomForestClassifier, GradientBoostingClassifier]
minio_endpoint = 'http://localhost:9000'
minio_access_key = 'QZYyGxSL5KANo5OnDdTb'
minio_secret_key = 'MehFcQ5G7ymCZEBr2QN0nsc3yHgZX8dJFvbF2ZfH'
minio_bucket_name = 'src'
minio_client = boto3.client('s3', endpoint_url=minio_endpoint, aws_access_key_id=minio_access_key,
                            aws_secret_access_key=minio_secret_key)


class Model:
    models: Dict[str, ModelsUnion] = {}

    def __init__(self, endpoint: str, access_key: str, secret_key: str):
        self.dvc = DVC(endpoint, access_key, secret_key)

    def train_model(self,
                    model_type: str,
                    params: dict,
                    features: List[List[int]],
                    labels: List[int]):
        """
        Train and push model to minio storage
        :param model_type: str
        :param params: dict
        :param features: List[List[int]]
        :param labels: List[int]
        :return: model type
        """
        if model_type == "random forest":
            model = RandomForestClassifier(**params)

        elif model_type == "gradient boosting":
            model = GradientBoostingClassifier(**params)

        else:
            return "you fool!"

        self.dvc.version_train_features(features)
        x_train, x_test, y_train, y_test = model_selection.train_test_split(
            features, labels, test_size=0.2, random_state=42,
        )

        model.fit(x_train, y_train)
        Model.models[model_type] = model

        buckets = [x['Name'] for x in minio_client.list_buckets()['Buckets']]
        if minio_bucket_name not in buckets:
            minio_client.create_bucket(Bucket=minio_bucket_name)

        object_name = f'{model_type}.joblib'
        with BytesIO() as f:
            joblib.dump(model, f)
            f.seek(0)
            minio_client.upload_fileobj(Bucket=minio_bucket_name, Key=object_name, Fileobj=f)

        return model_type

    @staticmethod
    def predict(model_type: str, features: List[List[int]]):
        """
        get model from minio and predict
        :param model_type: str
        :param features: List[List[int]]
        :return: prediction model
        """
        if model_type not in Model.models:
            return "model are not available"

        object_name = f'{model_type}.joblib'
        local_model_path = f'{model_type}.joblib'

        minio_client.download_file(Bucket=minio_bucket_name, Key=object_name, Filename=local_model_path)

        loaded_model = joblib.load(local_model_path)
        prediction = loaded_model.predict(features)
        return prediction.tolist()

    @staticmethod
    def delete_model(model_id: str):
        """
        delete model from minio storage
        :param model_id: str
        :return:
        """
        object_name = f'{model_id}.joblib'
        if Model.models[model_id]:
            minio_client.delete_object(Bucket=minio_bucket_name, Key=object_name)
        else:
            return "Error"
