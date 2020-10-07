# facial_expression_recognition

Developed a model using CNN and OpenCV which detects human face in real time through a web camera andpredicts whether a person is happy, angry,neutral,sad or surprised.

# Dataset 

Dataset was downloaded from Kaggle 

# Model

A deep CNN network is built using 8 convolutional layers. Then a flatten layer is used and Multilayer Perceptron was added at the end with a softmax activation function.

In file cv.py , model is loaded with the best performing weights and 'haarcascade_frontalface_default.xml' is imported using openCV. The code in the cv.py file is for detecting the faces , making rectangles around the faces and predicting the facial expression . 

# Implementation

Running cv.py file in terminal will automatically open the webcamera and in real time facial expression is predicted and displayed.




