from flask import Flask, render_template, url_for, request
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    v1 = request.form['v1']
    v2 = request.form['v2']
    amount = request.form['amount']
    user_input = [[v1, v2, amount]]
    user_input_scaled = scaler.transform(user_input)
    prediction = model.predict(user_input_scaled)[0]


    return render_template('predict.html',v1=v1, v2 = v2, amount = amount, predict = prediction)




if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)
 