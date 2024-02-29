from flask import Flask, jsonify, request, send_from_directory,send_file
# import tensorflowjs as tfjs
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tensorflow import keras
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from flask_cors import CORS
from tensorflow.python.framework import ops
ops.reset_default_graph()
app = Flask(__name__)
CORS(app)

# Load dataset
df = pd.read_csv('dataset.csv')


if 'Disorder' not in df.columns:
    raise ValueError("Column 'Disorder' not found in the dataset.")

# Extract features (X) and target variable (Y)
X = df.drop(columns=['Disorder'])
Y = df[['Disorder']]

# Apply label encoding to categorical variables in X
label_encoder_X = LabelEncoder()
X_encoded = X.apply(label_encoder_X.fit_transform)

# Apply one-hot encoding to the target variable (assuming Y is the target variable)
label_encoder_Y = LabelEncoder()
Y_encoded = Y.apply(label_encoder_Y.fit_transform)
y_one_hot = keras.utils.to_categorical(Y_encoded, num_classes=5)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_one_hot, test_size=0.2, random_state=42)

# Define the Keras model architecture
tf_model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(X_train.shape[1],)),  # Input shape based on the number of features
     keras.layers.Dense(100, activation='relu'),  # Example hidden layer
     keras.layers.Dense(5, activation='softmax')  # Output layer for multiclass classification with 5 classes
])

# Compile the model with categorical_crossentropy loss for multiclass classification
tf_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model using the one-hot encoded target variable
tf_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Save the model architecture to JSON file
tf_model_json = tf_model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(tf_model_json)

# Save the model weights
tf_model.save_weights('model_weights.h5')

#-------------------------DUMMY_DATA----------------------------------------
new_data_list = [0,0,0,0,0,0,1,1,1,1,1,1 , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

new_data_array = np.array(new_data_list)

# Reshape the array to match the input shape expected by the model
new_data_array = new_data_array.reshape(1, 24)  # Assuming one sample

# Predict using the loaded model
predictions = tf_model.predict(new_data_array)
print("############## PRED  ##############",np.argmax(predictions))

@app.route('/', methods = ['GET'])
def initServ():
    return 'Hello from server'
#MAHESH_----------------------------------------------------------------
# Endpoint for sending the initial model to the frontend
# @app.route('/get_model', methods=['GET'])
# def get_model_params():
#     # Serialize the global model parameters and send them to the frontend
#     model_params = global_model.get_params()
#     return jsonify(model_params)
# Endpoint for serving the TensorFlow.js model file
# @app.route('/get_tf_model', methods=['GET'])
# def get_tf_model():
#     return send_from_directory('.', 'model.json')
# Endpoint for receiving model updates from the frontend
# @app.route('/update_model', methods=['POST'])
# def update_model():
#     # Deserialize the model update received from the frontend
#     model_params_update = request.json
    
#     # Update the global model with the received model update
#     updated_model_params = {key: np.array(value) for key, value in model_params_update.items()}
#     global_model.set_params(**updated_model_params)
    
#     return jsonify({'message': 'Model updated successfully'})
#---------------------------------------------------------------------------------

#ABHAY CODE----------------------------------------------------------------------
@app.route('/get_model', methods=['GET'])
def get_model():
   
    #  # Serialize the TensorFlow model using pickle
    with open('tf_model.pkl', 'wb') as f:
        pickle.dump(tf_model, f)
    
    # Send the model file as a response
    return send_file('tf_model.pkl', as_attachment=True) 
    

@app.route('/download_weights', methods=['GET'])
def download_weights():
    # Send the weights file as a response
    return send_file('model_weights.pkl', as_attachment=True)
#----------------------------------------------------------------------------------

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
