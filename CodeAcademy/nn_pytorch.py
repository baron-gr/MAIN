import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

# import dataset into pandas dataframe
dataset_path = '/Users/bgracias/datasets/Car-Price-Data/car data.csv'
car_df = pd.read_csv(dataset_path)

# one hot encode categorical features
categorical_columns = ['Year', 'Fuel_Type', 'Transmission']
encoder = OneHotEncoder(sparse_output=False)
one_hot_encoded = encoder.fit_transform(car_df[categorical_columns])
one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))

# concat encoded and original dataframe
car_df_encoded = pd.concat([car_df, one_hot_df], axis=1)
car_df_encoded = car_df_encoded.drop(categorical_columns, axis=1)

# scale features using MinMax
scaler = MinMaxScaler()
car_df_encoded['kms_scaled'] = scaler.fit_transform(car_df_encoded[['Kms_Driven']])
car_df_encoded = car_df_encoded.drop(car_df_encoded[['Kms_Driven']], axis=1)

# split feature and target into torch tensors
features = car_df_encoded.columns[1:].to_list()
target = 'Selling_Price'

X = torch.tensor(car_df_encoded[features].values, dtype=torch.float)
y = torch.tensor(car_df_encoded[target].values, dtype=torch.float).view(-1,1)

# split data into test and train sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=42)

# create NN model
num_features = len(features)
hidden_1 = 128
hidden_2 = 256
hidden_3 = 256
hidden_4 = 128
hidden_5 = 64
hidden_6 = 32
output = 1

model = nn.Sequential(
    nn.Linear(num_features, hidden_1),
    nn.ReLU(),
    nn.Linear(hidden_1, hidden_2),
    nn.ReLU(),
    nn.Linear(hidden_2, hidden_3),
    nn.ReLU(),
    nn.Linear(hidden_3, hidden_4),
    nn.ReLU(),
    nn.Linear(hidden_4, hidden_5),
    nn.ReLU(),
    nn.Linear(hidden_5, hidden_6),
    nn.ReLU(),
    nn.Linear(hidden_6, output)
)

# initiate loss and optimiser
mse_loss = nn.MSELoss()
adam_optim = optim.Adam(model.parameters(), lr=0.001)

# train model
# epochs = 50000
# for epoch in range(epochs):
#     predictions = model(X_train)
#     train_loss = mse_loss(predictions, y_train)
#     train_loss.backward()
#     adam_optim.step()
#     adam_optim.zero_grad()
    
#     if (epoch+1) % 2500 == 0:
#         print(f'Epoch: {epoch+1}, training loss: {train_loss:.3f}')

# torch.save(model, 'model_20k.pth')

# evaluate model
trained_model = torch.load('model_20k.pth')
with torch.no_grad():
    preds = trained_model(X_test)
    test_loss = mse_loss(preds, y_test)
    # print(f'Testing loss: {test_loss:.3f}')
    
    # compare output values
    preds_arr = np.round(preds.numpy()[:, 0], 2)
    preds_df = pd.DataFrame({'predicted_price': preds_arr})
    y_df = pd.DataFrame({'actual_price': y_test.numpy()[:, 0]})
    combined_df = pd.concat([y_df, preds_df], axis=1)

# plot residuals
import matplotlib.pyplot as plt
combined_df['residual'] = combined_df['actual_price'] - combined_df['predicted_price']
plt.scatter(combined_df.index, combined_df['residual'], c='black', marker='.')
plt.vlines(x=combined_df.index, ymin=0, ymax=combined_df['residual'], colors='black', alpha=0.5)
plt.axhline(y=0, c='r')
plt.show()