import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from keras.models import load_model
import joblib

df = pd.read_csv('D:\ML\Fed ML\FML_MentalHealth_Backend\dataset.csv')

if 'Disorder' not in df.columns:
    raise ValueError("Column 'Disorder' not found in the dataset.")

X = df.drop(columns=['Disorder'])
Y = df[['Disorder']]

label_encoder_X = LabelEncoder()
X_encoded = X.apply(label_encoder_X.fit_transform)

label_encoder_Y = LabelEncoder()
Y_encoded = Y.apply(label_encoder_Y.fit_transform)
y_one_hot = keras.utils.to_categorical(Y_encoded, num_classes=5)


joblib.dump(label_encoder_X, 'label_encoder_X.pkl')
joblib.dump(label_encoder_Y, 'label_encoder_Y.pkl')

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_one_hot, test_size=0.2, random_state=42)
model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(X_train.shape[1],)),  # Input shape based on the number of features
    keras.layers.Dense(100, activation='relu'),  # Example hidden layer
    keras.layers.Dense(5, activation='softmax')  # Output layer for multiclass classification with 5 classes
])

model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train , y_train, epochs=1, validation_data=(X_test, y_test), verbose=0) 
model.save('model.keras')

# model = load_model('model.h5')
# if(model):
#     print(model.summary())
