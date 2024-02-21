import pickle
import os
import numpy as np

from flask import Flask, render_template, request

app = Flask(__name__)


# Load the model from a file
with open("model.pkl", 'rb') as file:
    loaded_model = pickle.load(file)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        data_dir = request.form.to_dict()
        data = list(data_dir.values())
        data = np.array(data).reshape(1, -1)
        prediction = loaded_model.predict(data)
        print(f"Data: {data}")
        print(f"Predction: {prediction[0]}")
        return "Data received"
    else:
        return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    data = [636.0, 2.0, 500.0, 289.0, 1.0, 0.0, 2.0,
            2.0, 1987.0, 5.0, 29.0, 3.0, 3.0, 0.0, 3.0, 0.0]
    data = np.array(data).reshape(1, -1)

    prediction = loaded_model.predict(data)

    return f"Prediction: {prediction[0]}"


if __name__ == '__main__':
    app.run(debug=True)
