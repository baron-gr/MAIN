import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

## Load & view the data
df = pd.read_csv('coaster_db.csv')
df.shape
df.head()
df.columns
df.dtypes
df.describe()

## Dropping single column
# df.drop(['Opening date'], axis=1)

## Subsetting the data
df = df[['coaster_name','Location', 'Status',
    'Manufacturer','year_introduced','latitude','longitude',
    'Type_Main','opening_date_clean','speed_mph','height_ft',
    'Inversions_clean','Gforce_clean']].copy()

## Redefining column data types
df['opening_date_clean'] = pd.to_datetime(df['opening_date_clean'])
df['year_introduced'] = pd.to_numeric(df['year_introduced'])

## Rename columns
df = df.rename(columns={'coaster_name':'Coaster_Name',
                   'year_introduced':'Year_Introduced',
                   'opening_date_clean':'Opening_Date',
                   'speed_mph':'Speed_mph',
                   'height_ft':'Height_ft',
                   'latitude':'Latitude',
                   'longitude':'Longitude',
                   'Inversions_clean':'Inversions',
                   'Gforce_clean':'Gforce'})

## Sum of missing/NULL values
df.isna().sum()

## Check for duplicate coaster names
df.loc[df.duplicated()]

## Checking example of duplicate
df.loc[df.duplicated(subset=['Coaster_Name'])]
df.query('Coaster_Name == "Crystal Beach Cyclone"')

## Removing duplicate and reset index
df = df.loc[~df.duplicated(subset=['Coaster_Name','Location','Opening_Date'])].reset_index(drop=True).copy()

## Count distinct
df['Year_Introduced'].value_counts()

## Distinct count of column
ax = df['Year_Introduced'].value_counts().head(10).plot(kind='bar', title='Top 10 Years of Coasters Introduced')
ax.set_xlabel('Year Introduced')
ax.set_ylabel('Count')

## Distribution of column
# Histogram
ax = df['Speed_mph'].plot(kind='hist', bins=20, title='Distribution of coaster speed (mph)')
ax.set_xlabel('Speed (mph)')

# Kernel density estimate
ax = df['Speed_mph'].plot(kind='kde', title='Distribution of coaster speed (mph)')
ax.set_xlabel('Speed (mph)')

## Scatter plot between two columns
df.plot(kind='scatter', x='Speed_mph', y='Height_ft', title='Coaster Speed vs. Height')

## Seaborn plots
sns.scatterplot(x='Speed_mph', y='Height_ft', hue='Year_Introduced', data=df)
sns.pairplot(df, vars=['Speed_mph', 'Height_ft', 'Year_Introduced', 'Gforce', 'Inversions'], hue='Type_Main')

## Pandas correlation & heatmaps
df_corr = df[['Speed_mph', 'Height_ft', 'Year_Introduced', 'Gforce', 'Inversions']].dropna().corr()
sns.heatmap(df_corr, annot=True)

## Location analysis: What are the locations with the fastest coasters (minimum of 10)?
ax = df.query('Location != "Other"') \
    .groupby('Location')['Speed_mph'] \
    .agg(['mean','count']) \
    .query('count >= 10') \
    .sort_values(by='mean')['mean'] \
    .plot(kind='barh', figsize=(12, 5), title= 'Average Coaster Speed (mph) by Location')
ax.set_xlabel('Average Coaster Speed (mph)')