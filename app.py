from flask import Flask, jsonify, request, send_from_directory
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)

# Load dataset
df = pd.read_csv('dataset.csv')

# Ensure that 'Disorder' column exists in the original dataset
if 'Disorder' not in df.columns:
    raise ValueError("Column 'Disorder' not found in the dataset.")

# Ensure that the target variable is not encoded already
if df['Disorder'].dtype != 'object':
    raise ValueError("Column 'Disorder' is not of type 'object' indicating it's not categorical.")

# Drop the target variable from the dataset
X = df.drop(columns=['Disorder'])

# Perform one-hot encoding on categorical variables
X_encoded = pd.get_dummies(X, drop_first=True)

# Extract the target variable
y = df['Disorder']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Initialize a basic model on the server
global_model = RandomForestClassifier(n_estimators=100, random_state=42)
global_model.fit(X_train, y_train)

tf_model = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(X_train.shape[1],)),  # Input shape based on the number of features
    keras.layers.Dense(100, activation='relu'),  # Example hidden layer
    keras.layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Load weights from scikit-learn model
tf_model.set_weights([...])  # Load weights from your trained scikit-learn model

# Save TensorFlow model to JSON file
tf_model_json = tf_model.to_json()  # Convert model to JSON format
with open('model.json', 'w') as json_file:
    json_file.write(tf_model_json)

# Endpoint for serving the TensorFlow.js model file
@app.route('/get_tf_model', methods=['GET'])
def get_tf_model():
    return send_from_directory('.', 'model.json')

@app.route('/', methods = ['GET'])
def initServ():
    return 'Hello from server'

# Endpoint for sending the initial model to the frontend
@app.route('/get_model', methods=['GET'])
def get_model_params():
    # Serialize the global model parameters and send them to the frontend
    model_params = global_model.get_params()
    return jsonify(model_params)

# Endpoint for receiving model updates from the frontend
@app.route('/update_model', methods=['POST'])
def update_model():
    # Deserialize the model update received from the frontend
    model_params_update = request.json
    
    # Update the global model with the received model update
    updated_model_params = {key: np.array(value) for key, value in model_params_update.items()}
    global_model.set_params(**updated_model_params)
    
    return jsonify({'message': 'Model updated successfully'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
