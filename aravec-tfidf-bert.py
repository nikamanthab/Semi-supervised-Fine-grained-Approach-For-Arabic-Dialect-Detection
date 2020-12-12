import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score,f1_score
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

'''
Preprocessing
'''
def preprocess_text(train_list, test_list):
    X_train_corrected_tweets = []
    for tweet in tqdm(train_list):
        new_tweet = re.findall( '[^A-Za-z:/_.0-9\\#@,=+\(\)]+' ,tweet)
        new_tweet = " ".join(new_tweet).replace('\xa0','').replace('\u200c','').replace('\U000fe329','').replace('\u2066','').replace('\u2069','').strip()
        X_train_corrected_tweets.append(new_tweet)

    X_dev_corrected_tweets = []
    for tweet in tqdm(test_list):
        new_tweet = re.findall( '[^A-Za-z:/_.0-9\\#@,=+\(\)]+' ,tweet) #[^\x00-\x19\x21-\x7F]+
        new_tweet = " ".join(new_tweet).replace('\xa0','').replace('\u200c','').replace('\U000fe329','').replace('\u2066','').replace('\u2069','').strip()
        X_dev_corrected_tweets.append(new_tweet)
    return X_train_corrected_tweets, X_dev_corrected_tweets

'''
Aravec model loading from gensim
Aravec model tuning using neural network model
'''
t_model = gensim.models.Word2Vec.load(
    '../downloads/aravec/full_uni_cbow_100_twitter/full_uni_cbow_100_twitter.mdl')

train_df = pd.read_csv('../NADI-2020_release_1.0/NADI_release/train_labeled.tsv',sep='\t')
dev_df = pd.read_csv('../NADI-2020_release_1.0/NADI_release/dev_labeled.tsv',sep='\t')

X_train_original,y_train = train_df["#2 tweet_content"],train_df["#3 country_label"]
X_dev_original,y_dev = dev_df["#2 tweet_content"],dev_df["#3 country_label"]
X_train_original, X_dev_original = preprocess_text(X_train_original, X_dev_original)


X_noov = []
for sentence in X_train_original:
    sentence = sentence.split(' ')
    new_sentence = []
    for word in sentence:
        if word in t_model.wv.vocab:
            new_sentence.append(word)
    X_noov.append(new_sentence)
X_noov_dev = []
for sentence in X_dev_original:
    sentence = sentence.split(' ')
    new_sentence = []
    for word in sentence:
        if word in t_model.wv.vocab:
            new_sentence.append(word)
    X_noov_dev.append(new_sentence)
    
X_train = []
counter = 0
for one_vec in X_noov:
    if one_vec == []:
        counter += 1
        one_vec = ['ومايشوف']
    word_vector = t_model.wv[ one_vec ]
#     word_vector = np.sum(word_vector,axis=0)
    word_vector = np.pad(word_vector,pad_width=((0,100-word_vector.shape[0]),(0,0)))
    X_train.append(word_vector)

X_dev = []
counter = 0
for one_vec in X_noov_dev:
    if one_vec == []:
        counter += 1
        one_vec = ['ومايشوف']
    word_vector = t_model.wv[ one_vec ]
#     word_vector = np.sum(word_vector,axis=0)
    word_vector = np.pad(word_vector,pad_width=((0,100-word_vector.shape[0]),(0,0)))
    X_dev.append(word_vector)

X_train = np.array(X_train)
X_dev = np.array(X_dev)

clf = MLPClassifier(hidden_layer_sizes=(512),max_iter=300)
clf.fit(X_train, y_train)
pred = clf.predict(X_dev)
print("Count vectorization + MLP:")
print(classification_report(y_dev, pred))

'''
multi BERT model loading and feature embedding extraction
Fine tuning of the features using neural network model
'''

model = SentenceTransformer('distiluse-base-multilingual-cased')
X_train = model.encode(X_train_original,show_progress_bar=True,batch_size=512)
X_train = np.array(X_train)
X_dev = model.encode(X_dev_original,show_progress_bar=True,batch_size=512)
X_dev = np.array(X_dev)

clf = MLPClassifier(hidden_layer_sizes=(512),max_iter=300)
clf.fit(X_train, y_train)
pred = clf.predict(X_dev)
print("Count vectorization + MLP:")
print(classification_report(y_dev, pred))


'''
TFIDF transform fitting and applying transformation to train,dev set
Training the neural network model for finetuning
'''
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train).toarray()
X_dev = vectorizer.transform(X_dev).toarray()

clf = MLPClassifier(hidden_layer_sizes=(512),max_iter=300)
clf.fit(X_train, y_train)
pred = clf.predict(X_dev)
print("Count vectorization + MLP:")
print(classification_report(y_dev, pred))

'''
Fasttext arabic model is loaded
neural network is trained
'''
from tqdm import tqdm_notebook
from nltk import word_tokenize
from pymagnitude import *


fasttext_model = Magnitude("../downloads/fasttext-arabic/fasttext-arabic.magnitude")
def fasttext(x):
    vectors = []
    for title in tqdm_notebook(x):
        vectors.append(np.average(fasttext_model.query(word_tokenize(title)), axis = 0))
    return np.array(vectors)

X_train = fasttext(X_train_original)
X_dev = fasttext(X_dev_original)

clf = MLPClassifier(hidden_layer_sizes=(512),max_iter=300)
clf.fit(X_train, y_train)
pred = clf.predict(X_dev)
print("Count vectorization + MLP:")
print(classification_report(y_dev, pred))
