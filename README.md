<!-- # facial_expression_recognition

Developed a model using CNN and OpenCV which detects human face in real time through a web camera andpredicts whether a person is happy, angry,neutral,sad or surprised.

# Dataset 

Dataset was downloaded from Kaggle 

# Model

A deep CNN network is built using 8 convolutional layers. Then a flatten layer is used and Multilayer Perceptron was added at the end with a softmax activation function.

In file cv.py , model is loaded with the best performing weights and 'haarcascade_frontalface_default.xml' is imported using openCV. The code in the cv.py file is for detecting the faces , making rectangles around the faces and predicting the facial expression . 

# Implementation

Running cv.py file in terminal will automatically open the webcamera and in real time facial expression is predicted and displayed.



 -->

 # Facial Expression Recognition

 This project uses **Convolutional Neural Networks (CNN)** and **OpenCV** to detect human faces in real-time via a webcam and classify facial expressions into five categories: **Happy, Angry, Neutral, Sad, or Surprised**.

 ## Features
- Detects faces in real-time using OpenCV.
- Predicts facial expressions using a deep learning model.
- Uses a pre-trained CNN model for high accuracy.

## Dataset
The dataset used for training the model was downloaded from **Kaggle**.

## Model Architecture
- The deep learning model consists of **8 convolutional layers**.
- A **Flatten layer** is used before the final classification layers.
- A **Multilayer Perceptron (MLP)** is added at the end.
- A **Softmax activation function** is used to classify facial expressions.

## Files in the Project
- `cv.py` - Loads the trained model, detects faces using OpenCV, and predicts facial expressions.
- `fer.py` - Contains code for training and evaluating the model.
- `Facial_expression_recognition.ipynb` - Jupyter notebook for training and testing the model.
- `Emotion_little_vgg.h5` - Pre-trained model weights.
- `haarcascade_frontalface_default.xml` - OpenCV's Haar cascade file for face detection.

## Installation & Dependencies
Ensure you have the following dependencies installed before running the project:

```bash
   pip install opencv-python keras tensorflow numpy matplotlib
```

## How to Run
To test the facial expression recognition in real-time, run:

```bash
   python cv.py
```

This will:

- Open the webcam.
- Detect human faces.
- Predict and display the facial expression in real-time.

## Acknowledgments

- The dataset used in this project is sourced from Kaggle.
- OpenCV’s Haar Cascade Classifier is used for face detection.

## Future Improvements

- Improve accuracy with more robust datasets.
- Implement more advanced deep learning techniques.
- Deploy the model as a web or mobile application.




