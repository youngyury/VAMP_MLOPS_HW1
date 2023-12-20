import os
import pickle


class DVC:
    def __init__(self, endpoint: str, access_key: str, secret_key: str):
        os.system("dvc init")
        os.system("dvc remote add -d minio s3://features -f")
        os.system(f"dvc remote modify minio endpointurl http://{endpoint}")
        os.system(f"dvc remote modify minio access_key_id {access_key}")
        os.system(f"dvc remote modify minio secret_access_key {secret_key}")

    @staticmethod
    def version_train_features(data):
        bytes = pickle.dumps(data)
        with open("train_features", 'wb') as f:
            f.write(bytes)
        os.system("dvc add train_features")
        os.system("git add train_features.dvc")
        os.system("git commit -m 'Add train features file with DVC'")
        os.system("git push")
        os.system("dvc push")
        os.system("rm train_features")