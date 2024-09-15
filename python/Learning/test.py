import os
import pandas as pd

dataset_path = '/Users/bgracias/Datasets'

files = []
for dirname, _, filenames in os.walk(dataset_path):
    for filename in filenames:
        if filename.endswith(".csv"):
            files.append(os.path.join(dirname, filename))

# for file in files:
#     df = pd.read_csv(file)
#     print(file[file.rindex('/')+1:])
#     print(df.shape)
#     print(df.head())

car_df = pd.read_csv(files[4])
car_df['Selling_Price'] = car_df['Selling_Price'].multiply(10000).astype(int)
car_df.groupby('Year').agg(
    avg_kms_driven = ('Kms_Driven', 'mean'),
    avg_selling_price = ('Selling_Price', 'mean')
).astype(int).reset_index()

def double_price(df, search_col, search_col_val, change_col, amount):
    df.loc[df[search_col] == search_col_val, change_col] *= amount
    return df

df = double_price(car_df, 'Year', 2012, 'Selling_Price', 1.2)

# print(car_df[car_df['Year'] == 2012])

print(car_df[car_df['Year'] == 2012][['Kms_Driven', 'Selling_Price']].apply(lambda x, y=2: x * y))