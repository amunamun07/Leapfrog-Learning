from Models.Iris_dataset_model import classify
from flask import Flask, jsonify

app = Flask(__name__)


@app.route('/prediction')
def prediction():
    sepal_len = 5.2
    sepal_wid = 3
    petal_len = 1.5
    petal_wid = 0.3
    vatiety = classify(sepal_len, sepal_wid, petal_len, petal_wid)
    return jsonify(prediction=vatiety)


if __name__ == '__main__':
    app.run()



