from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('house_price_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    med_inc = float(request.form['med_inc'])          # Median income
    house_age = float(request.form['house_age'])      # House age
    ave_rooms = float(request.form['ave_rooms'])      # Average rooms
    ave_bedrms = float(request.form['ave_bedrms'])    # Average bedrooms
    population = float(request.form['population'])    # Population

    # Predict
    features = np.array([[med_inc, house_age, ave_rooms, ave_bedrms, population]])
    prediction = model.predict(features)

    return render_template('index.html', prediction=f'Predicted House Price: ${prediction[0]*100000:.2f}')

if __name__ == '__main__':
    app.run(debug=True)