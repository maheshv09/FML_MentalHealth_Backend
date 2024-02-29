import React, { useState, useEffect } from "react";
import { View, Text, Button } from "react-native";
import * as tf from "@tensorflow/tfjs";
import { fetch } from "@tensorflow/tfjs-react-native";

//import { RandomForestClassifier } from "scikit-learn";
import axios from "axios";

const App = () => {
  const [questionNumber, setQuestionNumber] = useState(1);
  const [questions, setQuestions] = useState([]);
  const [answers, setAnswers] = useState({});
  const [model, setModel] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [modelParam, setModelParam] = useState(null);

  // useEffect(() => {
  //   // Fetch the TensorFlow.js model JSON file from the backend
  //   async function fetchModel() {
  //     setQuestions([
  //       "Do you often feel nervous?",
  //       "Do you experience panic attacks?",
  //       "Do you experience rapid breathing?",
  //       "Do you often sweat excessively?",
  //       "Do you have trouble concentrating?",
  //       "Do you have trouble sleeping?",
  //       "Do you have trouble with work or daily tasks?",
  //       "Do you often feel hopeless?",
  //       "Do you experience frequent anger or irritability?",
  //       "Do you tend to overreact to situations?",
  //       "Have you noticed a change in your eating habits?",
  //       "Have you experienced suicidal thoughts?",
  //       "Do you often feel tired or fatigued?",
  //       "Do you have close friends you can confide in?",
  //       "Do you spend excessive time on social media?",
  //       "Have you experienced significant weight gain or loss?",
  //       "Do you place a high value on material possessions?",
  //       "Do you tend to keep to yourself or prefer solitude?",
  //       "Do you frequently experience distressing memories?",
  //       "Do you have nightmares frequently?",
  //       "Do you tend to avoid people or activities?",
  //       "Do you often feel negative about yourself or your life?",
  //       "Do you have trouble concentrating or focusing?",
  //       "Do you often blame yourself for things?",
  //     ]);
  //     // const response = await fetch("http://localhost:5000/get_tf_model");
  //     // console.log("%%%%%%%%%%%%%  RESPONSE",response)
  //     // const modelJSON = await response.json();
  //     // const loadedModel = await tf.loadLayersModel(tf.io.fromMemory(modelJSON));
  //     // console.log("modelJSON :",modelJSON,"\n loadedModel :",loadedModel)
  //     // setModel(loadedModel);
      
  //     // tfjs.converters.save_keras_model(tf_model, tfjs_model_path)

  //     // const modelResponse = await fetch("http://localhost:5000/get_model");
  //     // console.log("modelRespnse :",modelResponse) 
  //     // const modelData = await modelResponse.json();
  //     // console.log("modelData :",modelData)
  //     // const model = await tf.loadLayersModel(tf.io.browserFiles([modelData.model_tfjs_json]));
  
  //     // const weightsResponse = await fetch("http://localhost:5000/download_weights", {
  //     //   responseType: 'arraybuffer'  // Tell fetch to expect binary data
  //     // });
  //     // const weightsBuffer = await weightsResponse.arrayBuffer();
  
  //     // // Deserialize the weights from the binary data
  //     // const weights = new Float32Array(weightsBuffer);
  //     // console.logs("Weights :",weights)
  //     // // Set the weights to the loaded model
  //     // model.setWeights([weights]);
  
  //     // // Set the loaded model
  //     // setModel(model);
  //     // const new_data_list = [0,0,0,0,0,0,1,1,1,1,1,1 , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
      
  //     // // Reshape the array to match the input shape expected by the model
  //     // const new_data_tensor = tf.tensor2d([new_data_list], [1, 24]);  // Assuming one sample
      
  //     // // Make prediction using the fetched model
  //     // const predictions = model.predict(new_data_tensor);
  //     // const predictionData = predictions.dataSync();
  //     // console.log("------------Prediction--------->>>>>>:", predictionData);   

  //     console.log("Hello 1")
  //     const modelResponse = await axios.get("http://localhost:5000/get_model", {
  //       responseType: "blob" // Set responseType to blob to receive binary data
  //     });
  //     console.log("Hello 2")
  //     const modelBlob = modelResponse.data;
  //     console.log("modelBlob",modelBlob)
  //     console.log("Hello 3")
  //     // Deserialize the model using TensorFlow.js
  //     const loadedModel = await tf.loadLayersModel(
  //       tf.io.fromMemory(await modelBlob.arrayBuffer())
  //     );
  //     console.log("Hello 4")
  //       console.log("LOADED MODEL :",loadedModel)
  //     setModel(loadedModel);
  //     //  const new_data_list = [0,0,0,0,0,0,1,1,1,1,1,1 , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
      
  //     // // Reshape the array to match the input shape expected by the model
  //     // const new_data_tensor = tf.tensor2d([new_data_list], [1, 24]);  // Assuming one sample
      
  //     // // Make prediction using the fetched model
  //     // const predictions = model.predict(new_data_tensor);
  //     // const predictionData = predictions.dataSync();
  //     // console.log("------------Prediction--------->>>>>>:", predictionData);   

  //    }
  //   fetchModel();
  // }, []);

  const handleAnswer = (question, answer) => {
    // Update local answers state
    setAnswers((prevAnswers) => ({ ...prevAnswers, [question]: answer }));

    // Move to the next question or submit answers if all questions answered
    if (questionNumber < questions.length) {
      setQuestionNumber(questionNumber + 1);
    } else {
      // Make prediction using local model
      //store prediction in jSon format
      console.log("Answers :",answers)
      // const localPrediction = predictWithLocalModel(model, answers);
      // setPrediction(localPrediction);

      // Send model updates to backend
      // axios
      //   .post("https://fml-mentalhealth-backend-1.onrender.com/update_model", {
      //     model_update: model,
      //   })
      //   .then((response) => {
      //     console.log("Model updates sent successfully");
      //   })
      //   .catch((error) => {
      //     console.error("Error sending model updates:", error);
      //   });
    }
  };

  // Function to make prediction using local model
  const predictWithLocalModel = async (model, answers) => {
    if (!model) {
        console.error("No model available for prediction");
        return null;
    }

    // Convert answers to the format expected by the model
    const formattedAnswers = Object.values(answers).map(answer => answer ? 1 : 0); // Convert boolean answers to 0 or 1
    console.log("Formatted Answers:", formattedAnswers);

    // Pad the formatted answers to ensure it has 24 features
    const paddedAnswers = formattedAnswers.concat(Array(24 - formattedAnswers.length).fill(0));
    console.log("Padded Answers:", paddedAnswers);

    // Convert the padded answers to a 2D tensor with shape [1, 24]
    const inputData = tf.tensor2d([paddedAnswers]);

    const predictions = model.predict(inputData);
    const predictionData = predictions.dataSync();
    console.log("PREDICT :",predictionData)
    return predictionData;
};

  return (
    <View>
      {questions.length > 0 && questionNumber <= questions.length ? (
        <View>
          <Text>{questions[questionNumber - 1]}</Text>
          <Button
            title="Yes"
            onPress={() => handleAnswer(questions[questionNumber - 1], true)}
          />
          <Button
            title="No"
            onPress={() => handleAnswer(questions[questionNumber - 1], false)}
          />
        </View>
      ) : (
        <View>
          <Text>Thank you for answering the questions.</Text>
          {prediction && <Text>Prediction: {prediction}</Text>}
        </View>
      )}
   
    </View>
  );
};

export default App;
