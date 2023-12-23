import uvicorn
from fastapi import FastAPI, Query
from src.params import (GBParams,
                         TrainingData,
                         RFParams,
                         ModelType,
                         PredictionData)
from src.manager import Model

app = FastAPI()
endpoint = "localhost:9000",
access_key = "minioadmin",
secret_key = "minioadmin"


@app.post("/train_model/")
async def train_model(params_rf: RFParams,
                      params_gb: GBParams,
                      data: TrainingData,
                      model_type: ModelType):
    """

    :param model_type: name of  model
    :param params_rf: hyperparameters for random forest
    :param params_gb: hyperparameters for gradient boosting
    :param data: labels and features
    :return: model id
    """
    model = Model(endpoint, access_key, secret_key)
    if model_type.model_type == "random forest":
        model_id = model.train_model(
            model_type=model_type.model_type,
            params=params_rf.dict(),
            features=data.features,
            labels=data.labels
        )

    elif model_type.model_type == "gradient boosting":
        model_id = model.train_model(
            model_type=model_type.model_type,
            params=params_gb.dict(),
            features=data.features,
            labels=data.labels
        )

    else:
        return "no modelz"

    return model_id


@app.get("/get_models")
def get_models():
    return {'models': list(Model.models.keys())}


@app.post("/predict")
def predict(data: PredictionData,
            model_id: str = Query(...)):
    return Model.predict(model_id, data.features)


@app.delete("/delete_model")
def delete_model(model_id: str = Query(...)):
    Model.delete_model(model_id)
