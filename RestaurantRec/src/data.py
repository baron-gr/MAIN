import sqlite3
import pandas as pd

conn = sqlite3.connect('restaurants.db')
df = pd.read_sql_query("SELECT count(*) FROM restaurants", conn)
conn.close()
print(df)
# print(df[df['neighbourhood'] == 'Holborn'].sort_values(by=['rating', 'num_reviews'], ascending=False))