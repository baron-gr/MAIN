import numpy as np
import pandas as pd
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

## 1--> Benign, 2 --> Malignant
# print(data_frame['label'].value_counts())
# print(data_frame.groupby('label').mean())

## Split data into Traing & Test sets
X = data_frame.drop(columns='label', axis=1)
Y = data_frame['label']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
# print(X.shape, X_train.shape, X_test.shape)

## Building the Neural Network
import tensorflow as tf