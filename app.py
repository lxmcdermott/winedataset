import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template

# load trained model
model = tf.keras.models.load_model('wine_quality_model.h5')

# init flask
app = Flask(__name__)

# define home page route
@app.route('/')
def home():
    return render_template('index.html')

# define predict route
@app.route('/predict', methods=['POST'])
def predict():
    # extract features from form data
    features = [float(x) for x in request.form.values()]
    features = np.array([features])

    # make predictions with model
    prediction = model.predict(features)
    
    return render_template('index.html', prediction_text=f'Predicted Value: {prediction[0][0]:.2f}')

if __name__ == "__main__":
    app.run(debug=True)
