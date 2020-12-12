import pandas as pd
import numpy as np
import re
from pymagnitude import *
from tqdm import tqdm
from nltk import word_tokenize

fast = Magnitude("../downloads/fasttext-arabic/fasttext-arabic.magnitude")
def input_preprocessor(df, type):
    new_tweets_arr = []
    for index in tqdm(df.index):
        tweet = df['#2 tweet_content'][index]
        tweet_id = df['#1 tweet_ID'][index]
        new_tweet = re.findall( '[^A-Za-z:/_.0-9\\#@,=+\(\)]+' ,tweet) #[^\x00-\x19\x21-\x7F]+
        new_tweet = " ".join(new_tweet).replace('\xa0','').replace('\u200c','').replace('\U000fe329','').replace('\u2066','').replace('\u2069','').strip()
        features = np.average(fast.query(word_tokenize(new_tweet)), axis = 0)
        np.savez_compressed('../features/'+type+'/'+str(tweet_id)+'.npz', features=features)
        
million = pd.read_csv('../tsv/10m_collected.tsv','\t')
million_available = million[million['#2 tweet_content'] != '<UNAVAILABLE>']
filtered_train = input_preprocessor(million_available,type='10million')