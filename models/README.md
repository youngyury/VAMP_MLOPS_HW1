# MLOPS HW1

This project is about deploy ml model using FastApi

## Installation

Before running, use venv to get dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
```

Now install the required dependencies:
```bash
pip install -r requirements.txt
```


## Running
After installation, you can run the FastAPI application using uvicorn:

```python
if __name__ == '__main__':
    uvicorn.run('main:app', port=8080, reload=True)
```

The application will be accessible at http://localhost:8080.

## Testing
For testing you can use FastApi docs http://localhost:8080/docs.

Or use CURL:

```bash
curl -X 'POST' \
  'http://127.0.0.1:8080/train_model/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "params_rf": {
    "n_estimators": 100,
    "max_depth": 10
  },
  "params_gb": {
    "n_estimators": 100,
    "max_depth": 10,
    "learning_rate": 0.1
  },
  "data": {
    "features": [
      [3.2, 2.1, 3.4, 1.2],
      [4.1, 3.2, 2.6, 0.7],
      [5.4, 4.1, 1.3, 1.2]
    ],
    "labels": [
      "z",
      "x",
      "c"
    ]
  }
}'
```
This will return accuracy of Random Forest model and accuracy of Gradient Boosting model.