from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np

## Instantiate model
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

## Encode and calculate sentiment
tokens = tokenizer.encode('It was good but couldve been better. Great', return_tensors='pt')
result = model(tokens)
check = int(torch.argmax(result.logits))+1

## Collect Reviews
rev = requests.get('https://www.yelp.com/biz/social-brew-cafe-pyrmont')
soup = BeautifulSoup(rev.text, 'html.parser')
regex = re.compile('.*comment.*')
results = soup.find_all('p', {'class': regex})
reviews = [result.text for result in results]

## Load reviews into dataframe and score
df = pd.DataFrame(np.array(reviews), columns=['review'])
df['review'].iloc[0]

def sentiment_score(review):
    tokens = tokenizer.encode(review, return_tensors='pt')
    result = model(tokens)
    return int(torch.argmax(result.logits))+1

sentiment_score(df['review'].iloc[1])

df['sentiment'] = df['review'].apply(lambda x: sentiment_score(x[:512]))
print(df['review'].iloc[3])
