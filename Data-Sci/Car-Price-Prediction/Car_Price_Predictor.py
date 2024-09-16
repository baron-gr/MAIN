import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics

car_data = pd.read_csv('/Users/bgracias/MAIN/Python/Data-Sci/Car-Price-Prediction/car data.csv')

## Investigate data
# print(car_data.head())
# print(car_data.shape)
# print(car_data.info())
# print(car_data.isnull().sum())
# print(car_data.Fuel_Type.value_counts())
# print(car_data.Seller_Type.value_counts())
# print(car_data.Transmission.value_counts())

## Encode categorical data
car_data.replace({'Fuel_Type':{'Petrol':0, 'Diesel':1, 'CNG':2}}, inplace=True)
car_data.replace({'Seller_Type':{'Dealer':0, 'Individual':1}}, inplace=True)
car_data.replace({'Transmission':{'Manual':0, 'Automatic':1}}, inplace=True)

## Splitting Data and making Target
X = car_data.drop(['Car_Name', 'Selling_Price'], axis=1)
Y = car_data['Selling_Price']

# print(X)
# print(Y)

## Splitting data into Test and Train
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=2)

## Model Training: Linear Regression
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train, Y_train)

## Model Evaluation: Training data
LR_training_data_prediction = lin_reg_model.predict(X_train)
LR_training_error_score = metrics.r2_score(Y_train, LR_training_data_prediction)
# print(LR_training_error_score)

## Visualise actual vs predicted prices
# plt.scatter(Y_train, LR_training_data_prediction)
# plt.xlabel("Actual Price")
# plt.ylabel("Predicted Price")
# plt.title("Actual vs Predicted Prices")
# plt.show()

## Model Evaluation: Testing data
LR_testing_data_prediction = lin_reg_model.predict(X_test)
LR_testing_error_score = metrics.r2_score(Y_test, LR_testing_data_prediction)
# print(LR_testing_error_score)

## Visualise actual vs predicted prices
# plt.scatter(Y_test, LR_testing_data_prediction)
# plt.xlabel("Actual Price")
# plt.ylabel("Predicted Price")
# plt.title("Actual vs Predicted Prices")
# plt.show()

## Model Training: Lasso Regression
lasso_model = Lasso()
lasso_model.fit(X_train, Y_train)

## Model Evaluation: Training data
L_training_data_prediction = lasso_model.predict(X_train)
L_training_error_score = metrics.r2_score(Y_train, L_training_data_prediction)
# print(L_training_error_score)

## Visualise actual vs predicted prices
# plt.scatter(Y_train, L_training_data_prediction)
# plt.xlabel("Actual Price")
# plt.ylabel("Predicted Price")
# plt.title("Actual vs Predicted Prices")
# plt.show()

## Model Evaluation: Testing data
L_testing_data_prediction = lasso_model.predict(X_test)
L_testing_error_score = metrics.r2_score(Y_test, L_testing_data_prediction)
# print(L_testing_error_score)

## Visualise actual vs predicted prices
# plt.scatter(Y_test, L_testing_data_prediction)
# plt.xlabel("Actual Price")
# plt.ylabel("Predicted Price")
# plt.title("Actual vs Predicted Prices")
# plt.show()