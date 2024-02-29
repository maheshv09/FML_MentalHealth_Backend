from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from keras.models import load_model
import joblib
from io import StringIO
import sys

app = Flask(__name__)
# df = pd.read_csv('D:\ML\Fed ML\FML_MentalHealth_Backend\dataset.csv')

# if 'Disorder' not in df.columns:
#     raise ValueError("Column 'Disorder' not found in the dataset.")

# X = df.drop(columns=['Disorder'])
# Y = df[['Disorder']]

# label_encoder_X = LabelEncoder()
# X_encoded = X.apply(label_encoder_X.fit_transform)

# label_encoder_Y = LabelEncoder()
# Y_encoded = Y.apply(label_encoder_Y.fit_transform)
# y_one_hot = keras.utils.to_categorical(Y_encoded, num_classes=5)

# X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_one_hot, test_size=0.2, random_state=42)
# model = keras.Sequential([
#     keras.layers.InputLayer(input_shape=(X_train.shape[1],)),  # Input shape based on the number of features
#     keras.layers.Dense(100, activation='relu'),  # Example hidden layer
#     keras.layers.Dense(5, activation='softmax')  # Output layer for multiclass classification with 5 classes
# ])

# model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
# model.fit(X_train , y_train, epochs=1, validation_data=(X_test, y_test), verbose=0)  



model = load_model('model.h5')
if(model):
    print(model.summary())
label_encoder_X = joblib.load('label_encoder_X.pkl')
label_encoder_Y = joblib.load('label_encoder_Y.pkl')


# List of questions

questions = [
    "Do you often feel nervous?",
    "Do you experience panic attacks?",
    "Do you experience rapid breathing?",
    "Do you often sweat excessively?",
    "Do you have trouble concentrating?",
    "Do you have trouble sleeping?",
    "Do you have trouble with work or daily tasks?",
    "Do you often feel hopeless?",
    "Do you experience frequent anger or irritability?",
    "Do you tend to overreact to situations?",
    "Have you noticed a change in your eating habits?",
    "Have you experienced suicidal thoughts?",
    "Do you often feel tired or fatigued?",
    "Do you have close friends you can confide in?",
    "Do you spend excessive time on social media?",
    "Have you experienced significant weight gain or loss?",
    "Do you place a high value on material possessions?",
    "Do you tend to keep to yourself or prefer solitude?",
    "Do you frequently experience distressing memories?",
    "Do you have nightmares frequently?",
    "Do you tend to avoid people or activities?",
    "Do you often feel negative about yourself or your life?",
    "Do you have trouble concentrating or focusing?",
    "Do you often blame yourself for things?"
]

# Column names for the DataFrame
columns = [
    "feeling.nervous", "panic", "breathing.rapidly", "sweating", "trouble.in.concentration",
    "having.trouble.in.sleeping", "having.trouble.with.work", "hopelessness", "anger", "over.react",
    "change.in.eating", "suicidal.thought", "feeling.tired", "close.friend", "social.media.addiction",
    "weight.gain", "material.possessions", "introvert", "popping.up.stressful.memory", "having.nightmares",
    "avoids.people.or.activities", "feeling.negative", "trouble.concentrating", "blamming.yourself"
]

# Initialize global DataFrame
global_df = pd.DataFrame(columns=columns)
# expanded_df=None

answerList=[]
@app.route('/', methods=['GET', 'POST'])
def question():
    global expanded_df
    if request.method == 'POST':
        # Save the answer to the global DataFrame
        answer = request.form.get('answer')
        answerList.append(answer)
        # Move to the next question or finish
        if len(answerList) < len(questions):
            return redirect(url_for('question'))
        else:
            global_df.loc[len(global_df)] = answerList
            expanded_df = pd.concat([global_df] * 1000, ignore_index=True)
            return redirect(url_for('finish'))

    question_text = questions[len(answerList)]
    return render_template('question.html', question=question_text, question_number=len(answerList)+1)

@app.route('/finish')
def finish():

   
    # global_df = global_df.append(pd.Series(answerList, index=global_df.columns), ignore_index=True)
    # expanded_df = global_df.loc[np.repeat(global_df.index, 100)]

    return render_template("finish.html" ,df=expanded_df,myanswers=answerList)

@app.route('/train_model/<myPred>', methods=['POST'])
def train_model(myPred):
    global expanded_df
    expanded_df['disorder']=myPred
    print(expanded_df.head(5))
    if 'disorder' not in expanded_df.columns:
        raise ValueError("Column 'disorder' not found in the dataset.")

    X = expanded_df.drop(columns=['disorder'])
    Y = expanded_df[['disorder']]

    X_encoded = X.apply(label_encoder_X.fit_transform)

    Y_encoded = Y.apply(label_encoder_Y.fit_transform)
    y_one_hot = keras.utils.to_categorical(Y_encoded, num_classes=5)
    
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_one_hot, test_size=0.2, random_state=42)
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(X_train.shape[1],)),  # Input shape based on the number of features
        keras.layers.Dense(100, activation='relu'),  # Example hidden layer
        keras.layers.Dense(5, activation='softmax')  # Output layer for multiclass classification with 5 classes
    ])

    model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train , y_train, epochs=1, validation_data=(X_test, y_test), verbose=0)  
    return render_template("train.html",df=expanded_df,modSum=model)


@app.route('/get_prediction', methods=['POST'])
def get_prediction():
    
    encoded_answers = label_encoder_X.transform(answerList)

    # Convert the list to a NumPy array
    new_data_array = np.array(encoded_answers)

    new_data_array = new_data_array.reshape(1, 24)  # Assuming one sample

    
    predictions = model.predict(new_data_array)
    predictions_1d = predictions.flatten()
    predicted_label = label_encoder_Y.inverse_transform([np.argmax(predictions_1d)])
    return render_template("pred.html",myPred=predicted_label[0],df=expanded_df)

if __name__ == '__main__':
    app.run(debug=True , use_reloader=False)
