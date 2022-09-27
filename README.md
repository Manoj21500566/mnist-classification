# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset

## DESIGN STEPS

### STEP 1:Import tensorflow and preprocessing libraries

### STEP 2:Build a CNN model

### STEP 3:Compile and fit the model and then predict

## PROGRAM
~~~

python3
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)

X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

np.unique(y_test)

model = keras.Sequential([
    tf.keras.layers.Conv2D(32,kernel_size=3,activation="relu",padding="same"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(32,kernel_size=3,activation="relu"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(32,kernel_size=3,activation="relu"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10,activation="softmax")
])

model.compile(loss="categorical_crossentropy", metrics='accuracy',optimizer="adam")

model.fit(X_train_scaled ,y_train_onehot, epochs=2,
          batch_size=64, 
          validation_data=(X_test_scaled,y_test_onehot))

pd.DataFrame(model.history.history).plot()

x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)

confusion_matrix(y_test,x_test_predictions)

print(classification_report(y_test,x_test_predictions))

img = image.load_img('imagefive.jpg')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0

np.argmax(model.predict(img_28_gray_scaled.reshape(1,28,28,1)),axis=1)
~~~

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![1out](https://user-images.githubusercontent.com/94588708/192436984-41b1803a-e9b4-4b27-b7fb-f5cb5ce22f9a.png)


### Classification Report

![2out](https://user-images.githubusercontent.com/94588708/192437010-6c4ffd4d-0405-4f04-9e04-b0877235d062.png)


### Confusion Matrix

![3out](https://user-images.githubusercontent.com/94588708/192437058-9bd8d08d-d893-4c5b-85fc-cb772779753e.png)


### New Sample Data Prediction

![4out](https://user-images.githubusercontent.com/94588708/192437160-07243918-fe15-403f-9cd3-9e74dd3ea899.png)

## RESULT
Thus a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is written and executed successfully.
