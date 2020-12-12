import numpy as np
import pandas as pd
import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.nn import Linear,Softmax,CrossEntropyLoss,Module,ReLU,DataParallel,Sequential
from torch.optim import Adam, SGD
from tqdm import tqdm_notebook
from nltk import word_tokenize
from pymagnitude import *
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from time import time
from sentence_transformers import SentenceTransformer
import csv
import warnings
warnings.filterwarnings('ignore')

device = 'cuda:1'

train_df = pd.read_csv('../tsv/final/oversampled_train.tsv','\t',names=['id','#3 country_label','a','#2 tweet_content'])
dev_df = pd.read_csv('../tsv/final/dev.tsv',sep='\t',names=['id','#3 country_label','a','#2 tweet_content'])

label_map = {}
y_train_original = train_df["#3 country_label"]
for u in range(len(y_train_original.unique())):
    label_map[y_train_original.unique()[u]] = int(u)
reverse_label_map = {value : key for (key, value) in label_map.items()}
def label_onehot(label):
    onehot = np.zeros((21))
    index = label_map[label]
    onehot[index] = 1
    return onehot

fast = Magnitude("../downloads/fasttext-arabic/fasttext-arabic.magnitude")
def fasttext(x):
    vectors = []
    for title in x:
        vectors.append(np.average(fast.query(word_tokenize(title)), axis = 0))
    return np.array(vectors)

class ArabicDataset(Dataset):
    def __init__(self, csv_file=None, million_csv=None, transform=None):
        fast = Magnitude("../downloads/fasttext-arabic/fasttext-arabic.magnitude")
        def transform(x):
            vectors = []
            for title in tqdm(x):
                vectors.append(np.average(fast.query(word_tokenize(title)), axis = 0))
            return np.array(vectors)
        
        self.csv_file = csv_file
        if csv_file:         
            if million_csv:
                self.text_df = pd.concat([pd.read_csv(csv_file, sep='\t'),
                                         pd.read_csv(million_csv, sep='\t')])
            else:
                self.text_df = pd.read_csv(csv_file,sep='\t')
        else:
            self.text_df = pd.read_csv(million_csv,sep='\t')
#         self.fasttext_data = transform(self.text_df['#2 tweet_content'])
    def __len__(self):
        return len(self.text_df)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        text = self.text_df.iloc[idx]['#2 tweet_content']
        content = text
#         text = self.fasttext_data[idx]
        text = fasttext([text]).reshape((300))
        if self.csv_file:
            sample = {'text': torch.from_numpy(text).to(device), 
                      'id': self.text_df.iloc[idx]['#1 tweet_ID']
                     }
        else:
            sample = {'text':text}
        return sample
    
    
class TuningNet(Module):
    def __init__(self, D_in, H,D_out):
        super(TuningNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear1_1 = torch.nn.Linear(H, H)
        self.linear2 = torch.nn.Linear(H, D_out)
        self.relu = ReLU()
        self.softmax = Softmax(dim=1)

    def forward(self, x):
        x = self.linear1(x)
#         x = self.relu(x)
#         x = self.linear1_1(x)
        x = self.relu(x)
        x = self.linear2(x)
        y_pred = self.softmax(x)
        return y_pred
    
train_csv_path = '../tsv/added_train.tsv'
dev_csv_path = '../tsv/final/test.tsv'
million_csv_path = '../semi_supervised_train/tsv_iters/iter1.tsv'
num_of_epochs = 50
learning_rate = 0.001
model = TuningNet(300,512,21).to(device)
train_batch_size = 32


model = torch.load('../semi-supervised_train/models/task1-sgd-lvl2-2000.pt').to(device)

# traindataset = ArabicDataset(train_csv_path) #,million_csv_path)
# trainloader = DataLoader(traindataset, batch_size=train_batch_size,
#                         shuffle=True)
# testloader = DataLoader(traindataset, batch_size=1000)
devdataset = ArabicDataset(dev_csv_path)
devloader = DataLoader(devdataset, batch_size=1)

criterian = CrossEntropyLoss().to(device)
# optimizer = SGD(model.parameters(), lr=learning_rate,momentum=0.9)
optimizer = Adam(model.parameters(), lr=learning_rate)
train_f1 = []
dev_f1 = []
y_pred = []
y_true = []
for epoch in range(1):
#     y_pred = []
#     y_true = []
#     with open('../tsv/dev_pred.txt','w') as f:
#         for batch in tqdm(devloader):
#             output = model(batch['text'])
#             index = output.argmax(dim=1).detach().cpu().numpy()[0]
#             f.write(reverse_label_map[index]+'\n')
#     #         y_pred += list(output.argmax(dim=1).detach().cpu().numpy())
#     #         y_true += list(batch['label'].argmax(dim=1).detach().cpu().numpy())
    f = open('../tsv/test_pred-task1.txt','wt')
    with open('../tsv/test_pred-task1.tsv', 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(['#1 tweet_ID', '#3 country_label', 'conf'])
        
        for batch in tqdm(devloader):
            output = model(batch['text'].to(device))
            index = output.argmax(dim=1).detach().cpu().numpy()[0]
#             out_file.write(reverse_label_map[index]+'\n')

            f.write(reverse_label_map[index]+'\n')
            tsv_writer.writerow([str(batch['id'][0]),
                    reverse_label_map[index],
                    str(float(output[0][index].detach().cpu().numpy()))])
