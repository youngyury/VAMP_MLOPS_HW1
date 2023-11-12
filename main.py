import uvicorn
from fastapi import FastAPI
from models.models import predict_model
from models.params import GBParams, TrainingData, RFParams


app = FastAPI()


@app.post("/train_model/")
async def train_model(params_rf: RFParams,
                      params_gb: GBParams,
                      data: TrainingData):
    """

    :param params_rf: hyperparameters for random forest
    :param params_gb: hyperparameters for gradient boosting
    :param data: labels and features
    :return:
    """
    accuracy_gb, accuracy_rf = predict_model(
        params_rf=params_rf.dict(),
        params_gb=params_gb.dict(),
        features=data.features,
        labels=data.labels,
    )
    return {'accuracy_random_forest': accuracy_rf, 'accuracy_gradboosting': accuracy_gb}

if __name__ == '__main__':
    uvicorn.run('main:app', port=8080, reload=True)

