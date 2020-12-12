import pandas as pd
import numpy as np
import re
from pymagnitude import *
from tqdm import tqdm
from nltk import word_tokenize
import os

'''
Loading fasttext model, Cleaning text and saving the np arrays
'''
fast = Magnitude("../downloads/fasttext-arabic/fasttext-arabic.magnitude")
error_tweet_id = 0
error_tweet = ''
def input_preprocessor(df, type):
    new_tweets_arr = []
    for index in tqdm(df.index):
        tweet = df['#2 tweet_content'][index]
        tweet_id = df['#1 tweet_ID'][index]
        error_tweet_id = tweet_id
        error_tweet = tweet
        new_tweet = re.findall( '[^A-Za-z:/_.0-9\\#@,=+\(\)]+' ,tweet) #[^\x00-\x19\x21-\x7F]+
        new_tweet = " ".join(new_tweet).replace('\xa0','').replace('\u200c','').replace('\U000fe329','').replace('\u2066','').replace('\u2069','').strip()
        features = np.average(fast.query(word_tokenize(new_tweet)), axis = 0)
        np.savez_compressed('../features/'+type+'/'+str(tweet_id)+'.npz', features=features)
        
train = pd.read_csv('../NADI-2020_release_1.0/NADI_release/train_labeled.tsv','\t')
filtered_train = input_preprocessor(train,type='train')
print(len(os.listdir('../features/train/')))
dev = pd.read_csv('../NADI-2020_release_1.0/NADI_release/dev_labeled.tsv','\t')
filtered_train = input_preprocessor(dev,type='dev')
print(len(os.listdir('../features/dev/')))
test = pd.read_csv('../NADI-2020_TEST_2.0/NADI-2020_TEST_2.0/test_unlabeled.tsv','\t')
filtered_train = input_preprocessor(test,type='test')
print(len(os.listdir('../features/test/')))

'''
10 Million dataset feature extraction
'''
def input_preprocessor(df, type):
    new_tweets_arr = []
    for index in tqdm(df.index):
        tweet = df['#2 tweet_content'][index]
        tweet_id = df['#1 tweet_ID'][index]
        new_tweet = re.findall( '[^A-Za-z:/_.0-9\\#@,=+\(\)]+' ,tweet) #[^\x00-\x19\x21-\x7F]+
        new_tweet = " ".join(new_tweet).replace('\xa0','').replace('\u200c','').replace('\U000fe329','').replace('\u2066','').replace('\u2069','').strip()
        new_tweets_arr.append(new_tweet)
    return new_tweets_arr
def fasttext(df, type):
    for index in tqdm(df.index):
        new_tweet = df['arabic'][index]
        tweet_id = df['#1 tweet_ID'][index]
        features = np.average(fast.query(word_tokenize(new_tweet)), axis = 0)
        np.savez_compressed('../features/'+type+'/'+str(tweet_id)+'.npz', features=features)
        
million = pd.read_csv('../tsv/10m_collected.tsv','\t')
million_available = million[million['#2 tweet_content'] != '<UNAVAILABLE>']
filtered_train = input_preprocessor(million_available,type='10million')
million_available['arabic'] = filtered_train
million_available = million_available[million_available['arabic'] != '']
print(len(million_available))
fasttext(million_available,type='10million')

million_available[['#1 tweet_ID','#2 tweet_content']].to_csv('../tsv2/million_task1.tsv','\t',index=False)