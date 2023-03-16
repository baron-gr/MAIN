import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
from transformers import pipeline

plt.style.use('ggplot')

## Read in data
df = pd.read_csv('Reviews.csv')
df = df.head(500)
# print(df.shape)

## Quick EDA
ax = df['Score'].value_counts().sort_index(). \
    plot(kind='bar',
         figsize=(10, 5),
         title='Count of Reviews by Stars')
ax.set_xlabel('Review Stars')
# plt.show()

## Basic NLTK
example = df['Text'][50]
tokens = nltk.word_tokenize(example)
tagged = nltk.pos_tag(tokens)
entities = nltk.chunk.ne_chunk(tagged)
# print(entities)

## VADAR - Valence Aware Dictionary and sEntiment Reasoner
sia = SentimentIntensityAnalyzer()
test = sia.polarity_scores('I am so happy!')
test2 = sia.polarity_scores('I am so sad!')
example_test = sia.polarity_scores(example)
# print(test)
# print(test2)
# print(example_test)

## Run polarity analysis on entire dataset
result = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['Text']
    myid = row['Id']
    result[myid] = sia.polarity_scores(text)

## Sentiment score & metadata
vaders = pd.DataFrame(result).T
vaders = vaders.reset_index().rename(columns={'index': 'Id'})
vaders = vaders.merge(df, how='left')

## Plot VADER results
ax = sns.barplot(data=vaders, x='Score', y='compound')
ax.set_title('Compound Score by Amazon Review Stars')
# plt.show()

fig, axs = plt.subplots(1, 3, figsize=(15, 3))
sns.barplot(data=vaders, x='Score', y='pos', ax=axs[0])
sns.barplot(data=vaders, x='Score', y='neu', ax=axs[1])
sns.barplot(data=vaders, x='Score', y='neg', ax=axs[2])
axs[0].set_title('Positive Score by Amazon Review Stars')
axs[1].set_title('Neutral Score by Amazon Review Stars')
axs[2].set_title('Negative Score by Amazon Review Stars')
plt.tight_layout()
# plt.show()

## Roberta Pretrained Model
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
 
## Run for Roberta Pretrained Model
def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2]
    }
    return scores_dict

## Combine VADER and Roberta models
result = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:   
        text = row['Text']
        myid = row['Id']
        # VADER results
        vader_result = sia.polarity_scores(text)
        vader_result_rename = {}
        for key, value in vader_result.items():
            vader_result_rename[f"vader_{key}"] = value
        # Roberta results
        roberta_result = polarity_scores_roberta(text)
        both = {**vader_result_rename, **roberta_result}
        result[myid] = both
    except RuntimeError:
        print(f"Error for id: {row['Id']}")

## Enter combined results into dataframe
results_df = pd.DataFrame(result).T
results_df = results_df.reset_index().rename(columns={'index': 'Id'})
results_df = results_df.merge(df, how='left')

## Compare Scores between VADER and Roberta and compare
sns.pairplot(data=results_df,
             vars=['vader_neg', 'vader_neu', 'vader_pos',
                  'roberta_neg', 'roberta_neu', 'roberta_pos'],
             hue='Score',
             palette='tab10')
# plt.show()

## Review Examples: False positives i.e. positive 1 star reviews
FP_roberta = results_df.query('Score == 1').sort_values('roberta_pos', ascending=False)['Text'].values[0]
FP_vader = results_df.query('Score == 1').sort_values('vader_pos', ascending=False)['Text'].values[0]
# print(FP_roberta)
# print(FP_vader)

## Review Examples: False negatives i.e. negative 5 star reviews
FN_roberta = results_df.query('Score == 5').sort_values('roberta_neg', ascending=False)['Text'].values[0]
FN_vader = results_df.query('Score == 5').sort_values('vader_neg', ascending=False)['Text'].values[0]
# print(FN_roberta)
# print(FN_vader)

## Transformers Pipeline usage
sent_pipeline = pipeline("sentiment-analysis")
test = sent_pipeline('I love sentiment analysis')