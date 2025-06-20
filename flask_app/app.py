
from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Define the expected form fields
        fields = [
    'area', 'bedrooms', 'bathrooms', 'stories', 
    'mainroad_yes', 'guestroom_yes', 'basement_yes',
    'hotwaterheating_yes', 'airconditioning_yes', 'parking', 
    'prefarea_yes' ]



        # Fetch and convert input values
        features = []
        for field in fields:
            value = request.form.get(field)
            if value is None or value == "":
                return render_template('index.html', prediction_text="Please fill all the fields.")
            features.append(float(value))

        # Transform input and make prediction
        final_input = scaler.transform([features])
        prediction = model.predict(final_input)[0]

        return render_template('index.html',
                               prediction_text=f'Predicted house price: ₹{int(prediction):,}')
    
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
