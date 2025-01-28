import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

## Import dataset as pandas dataframe
cc_data = pd.read_csv('/Users/bgracias/MAIN/Python/Data-Sci/Credit-Card-Fraud-Detection/creditcard.csv')

## View dataset
# print(cc_data.head())
# print(cc_data.tail())

## Dataset information
# print(cc_data.info())

## Numbert of missing values
# print(cc_data.isnull().sum())

## Dist of legit vs fraud transactions
# print(cc_data['Class'].value_counts())

## This dataset is highly unbalanced
## 0 --> Legit transaction
## 1 --> Fraud transaction

## Separating data for analysis
legit = cc_data[cc_data.Class == 0]
fraud = cc_data[cc_data.Class == 1]
# print(legit.shape)
# print(fraud.shape)

## Statistical measures of the data
# print(legit.Amount.describe())
# print(fraud.Amount.describe())

## Compare values for both transaction types
# print(cc_data.groupby('Class').mean())

## Under-Sampling
## Build sample dataset containing similar distribution of legit and fraud transactions
## No. Fraud = 492
legit_sample = legit.sample(n=492)

## Concat two DFs
new_dataset = pd.concat([legit_sample, fraud], axis=0)

# print(new_dataset.head())
# print(new_dataset['Class'].value_counts())
# print(new_dataset.groupby('Class').mean())

## Split the dataset into Features and Targets
X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']
# print(X)
# print(Y)

## Split data into Training & Testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
# print(X.shape, X_train.shape, X_test.shape)

## Model Training: Logistic Regression
model = LogisticRegression()

## Training Logistic Regression Model with Training Data
model.fit(X_train, Y_train)

## Model Evaluation: Accuracy Score on Training Data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print(training_data_accuracy)

## Model Evaluation: Accuracy Score on Testing Data
X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print(testing_data_accuracy)