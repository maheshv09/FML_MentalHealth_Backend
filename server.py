from flask import Flask, jsonify, request, send_file    
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import pickle
app = Flask(__name__)
import base64
# Placeholder for the server model
server_model = None

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
print("HII" + str(X_train.shape))
print(y_train.shape)
# Train the model using the one-hot encoded target variable
tf_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Save the model architecture to JSON file
tf_model_json = tf_model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(tf_model_json)

# Save the model weights
tf_model.save_weights('model_weights.h5')




@app.route('/', methods=['GET'])
def initServ():
    return "HELLO SERVER!!"

# @app.route('/get_model', methods=['GET'])
# def get_model():
#     global server_model
    
#     # Load the dataset
#     df = pd.read_csv('dataset.csv')

#     # Prepare data
#     X = df.drop(columns=['Disorder'])
#     Y = df['Disorder']

#     # Encode target variable
#     label_encoder_Y = LabelEncoder()
#     Y_encoded = label_encoder_Y.fit_transform(Y)

#     # Split the data into train and test sets
#     X_train, X_test, y_train, y_test = train_test_split(X, Y_encoded, test_size=0.2, random_state=42)

#     # Build and train the model
#     model = Sequential([
#         Dense(100, input_dim=X_train.shape[1], activation='relu'),
#         Dense(5, activation='softmax')
#     ])
#     model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#     model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

#     # Serialize the model to JSON
#     model_json = model.to_json()

#     # Save the model as the server model
#     global server_model
#     server_model = model

#     # Return model architecture and weights
#     return jsonify(model=model_json, weights=model.get_weights())

@app.route('/get_model', methods=['GET'])
def get_model():
    # Assuming tf_model is your trained TensorFlow model
    # Serialize the model architecture to JSON
    model_json = tf_model.to_json()

    # Convert the model weights to binary data
    weights_data = tf_model.get_weights()

    # Encode the weights data using base64
    encoded_weights = [base64.b64encode(w).decode('utf-8') for w in weights_data]

    # Return the model architecture and weights
    return jsonify(model=model_json, weights=encoded_weights)


@app.route('/get_weights', methods=['GET'])
def get_weights():
    return send_file('model_weights.h5', as_attachment=True)

@app.route('/update_model', methods=['POST'])
def update_model():
    global server_model

    # Receive updated model weights from the client
    updated_weights = request.json['weights']

    # Update the server model with received weights
    server_model.set_weights(updated_weights)

    return 'Model updated successfully!'

if __name__ == '__main__':
    # Start the Flask application
    app.run(debug=True)
