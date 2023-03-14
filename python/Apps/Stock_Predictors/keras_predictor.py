import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

# Set up
start = '2010-01-01'
end = '2019-12-31'

st.title('Stock Price Prediction')
user_input = st.text_input('Enter a stock symbol', 'AAPL')
df = data.DataReader(user_input, 'yahoo', start, end)

# Describing Data
st.subheader('Data from 2010 - 2019')
st.write(df.describe())

# Visualisations
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA')
ma100 = df.Close.rolling(window=100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, 'r')
plt.plot(df.Close, 'b')
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 200MA')
ma200 = df.Close.rolling(window=200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma100 = df.Close.rolling(window=100).mean()
ma200 = df.Close.rolling(window=200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)

# Splitting data into train and test set
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

scalar = MinMaxScaler(feature_range=(0, 1))
data_training = scalar.fit_transform(data_training)

# Splitting data into x_train and y_train sets
x_train = []
y_train = []

for i in range(100, data_training.shape[0]):
    x_train.append(data_training[i-100: i])
    y_train.append(data_training[i, 0])

x_train = np.array(x_train)
y_train = np.array(y_train)

# ML model
model = Sequential()
model.add(LSTM(units = 50, activation = 'relu', return_sequences = True, input_shape = (x_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units = 60, activation = 'relu', return_sequences = True))
model.add(Dropout(0.3))

model.add(LSTM(units = 80, activation = 'relu', return_sequences = True))
model.add(Dropout(0.4))

model.add(LSTM(units = 120, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(units = 1))
model.compile(optimizer = 'adam', loss ='mean_squared_error')
model.fit(x_train, y_train, epochs = 50)

# Save and load model
model.save('keras_model.h5')
model = load_model('keras_model.h5')

# Testing
past_100_days = data_training.tail(100)
final_df  = past_100_days.append(data_testing, ignore_index=True)
input_data = scalar.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test = np.array(x_test)
y_test = np.array(y_test)
y_predicted = model.predict(x_test)
scalar = scalar.scale_

scale_factor = 1/scalar[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Final graph
st.subheader('Stock Price Prediction')
fig = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label = 'Actual Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)