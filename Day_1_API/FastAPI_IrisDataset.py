from Models.Iris_dataset_model import classify
from fastapi import FastAPI

app = FastAPI()


@app.get('/prediction')
def prediction():
    sepal_len = 5.2
    sepal_wid = 3
    petal_len = 1.5
    petal_wid = 0.3
    vatiety = classify(sepal_len, sepal_wid, petal_len, petal_wid)
    return {"prediction": vatiety}


# to run this use "uvicorn FastAPI_risDataset:app --reload'