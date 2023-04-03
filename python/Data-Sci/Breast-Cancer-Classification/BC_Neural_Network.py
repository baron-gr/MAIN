import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

breast_cancer_dataset = sklearn.datasets.load_breast_cancer()
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns = breast_cancer_dataset.feature_names)
data_frame['label'] = breast_cancer_dataset.target

## Data investigation
# print(data_frame.shape)
# print(data_frame.info())
# print(data_frame.isnull().sum())
# print(data_frame.describe())

## 1 --> Benign, 0 --> Malignant
# print(data_frame['label'].value_counts())
# print(data_frame.groupby('label').mean())

## Split data into Traing & Test sets
X = data_frame.drop(columns='label', axis=1)
Y = data_frame['label']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
# print(X.shape, X_train.shape, X_test.shape)

## Standardise the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

## Building the Neural Network
import tensorflow as tf
tf.random.set_seed(3)
from tensorflow import keras

## Setting up the neural network layers
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(30,)),
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dense(2, activation='sigmoid')
])

## Compiling the neural network
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

## Training the neural network
history = model.fit(X_train_std, Y_train, validation_split=0.1, epochs=10)

## Visualizing accuracy and loss
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('epochs')
plt.legend(['training data', 'validation data'], loc='lower right')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('epochs')
plt.legend(['training data', 'validation data'], loc='upper right')
# plt.show()

## Accuracy of the model on test data
loss, accuracy = model.evaluate(X_test_std, Y_test)
# print(X_test_std.shape)
# print(X_test_std[0])

## Prediction probability of each class for that data point
Y_pred = model.predict(X_test_std)
# print(Y_pred.shape)
# print(Y_pred[0])
# print(X_test_std)
# print(Y_pred)

## Converting prediction probabilities to class labels
Y_pred_labels = [np.argmax(i) for i in Y_pred]
# print(Y_pred_labels)

## Building the predictive system
## Malignant
input_data = (20.57,17.77,132.9,1326,0.08474,0.07864,0.0869,0.07017,0.1812,0.05667,0.5435,0.7339,3.398,74.08,0.005225,0.01308,0.0186,0.0134,0.01389,0.003532,24.99,23.41,158.8,1956,0.1238,0.1866,0.2416,0.186,0.275,0.08902)
## Benign
# input_data = (13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259)

input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
input_data_std = scaler.transform(input_data_reshaped)

prediction = model.predict(input_data_std)
# print(prediction)

prediction_label = [np.argmax(prediction)]
# print(prediction_label)

if(prediction_label[0] == 0):
    print('The tumour is Malignant')
else:
    print('The tumour is Benign')