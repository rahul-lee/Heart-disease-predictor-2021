from sklearn.preprocessing import MinMaxScaler
import numpy as np
from flask import Flask, request, render_template
import joblib as jb

app = Flask(__name__)

model = jb.load('heart_predict.joblib')

X = [[65.0, 1.0, 3.0, 145, 233.0, 1.0, 0.0, 150, 0.0,
      2.3, 0.0, 0.0, 1.0]]
sc = MinMaxScaler(feature_range=(0, 1))
dataset_scaled = sc.fit_transform(X)


@app.route('/')
def home():
    return render_template('heart.html')


@app.route('/predict', methods=['GET', 'POST'])
def predictHeart():
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict(sc.transform(final_features))

    if prediction == 1:
        pred = "You have a Heart disease!"
    elif prediction == 0:
        pred = "You don't have a Heart disease."
    output = pred
    return render_template('heart.html', predicted='{}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
