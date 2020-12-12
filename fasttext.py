import numpy as np
import pandas as pd
import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.nn import Linear,Softmax,CrossEntropyLoss,Module,ReLU,DataParallel,Sequential
from torch.optim import Adam
from tqdm import tqdm_notebook
from nltk import word_tokenize
from pymagnitude import *
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from time import time
from sentence_transformers import SentenceTransformer
device = 'cuda:1'

train_df = pd.read_csv('../tsv2/train_task1.tsv','\t')

label_map = {}
y_train_original = train_df["#3 country_label"]
keykey = list(y_train_original.unique())
# keykey.reverse()
for u in range(len(keykey)):
    label_map[keykey[u]] = int(u)
reverse_label_map = {value : key for (key, value) in label_map.items()}
def label_onehot(label):
    onehot = np.zeros((21))
    index = label_map[label]
    onehot[index] = 1
    return onehot

class ArabicDataset(Dataset):
    def __init__(self, csv_file=None, million_csv=None, transform=None):
        fast = Magnitude("../downloads/fasttext-arabic/fasttext-arabic.magnitude")
        def transform(x):
            vectors = []
            for title in tqdm(x):
                vectors.append(np.average(fast.query(word_tokenize(str(title))), axis = 0))
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
        self.fasttext_data = transform(self.text_df['#2 tweet_content'])
    def __len__(self):
        return len(self.text_df)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        text = self.fasttext_data[idx]
#         text = fasttext([text]).reshape((300))
        if self.csv_file:
            label = 'Iraq'
            label = label_onehot(label)
            sample = {'text': text, 
                      'label': torch.from_numpy(label).to(device)}
        else:
            sample = {'text':text}
        return sample
    
class TuningNet(Module):
    def __init__(self, D_in, H,D_out):
        super(TuningNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.relu = ReLU()
        self.linear2 = torch.nn.Linear(H, D_out)
        self.softmax = Softmax(dim=1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        y_pred = self.softmax(x)
        return y_pred
    
train_csv_path = '../tsv/task1-lvl2-2000_train.tsv'
dev_csv_path = '../tsv/final/test.tsv'
million_csv_path = '../semi_supervised_train/tsv_iters/iter1.tsv'
num_of_epochs = 50
learning_rate = 0.001
model = TuningNet(300,512,21).to(device)
train_batch_size = 32

learning_rate=0.001
# model = torch.load('../semi-supervised_train/models/task1-sgd-lvl2-2000.pt').to(device)
model = TuningNet()

traindataset = ArabicDataset(train_csv_path)
trainloader = DataLoader(traindataset, batch_size=32,shuffle=True)
# testloader = DataLoader(traindataset, batch_size=1000)
devdataset = ArabicDataset(dev_csv_path)
devloader = DataLoader(devdataset, batch_size=1000)

criterian = CrossEntropyLoss().to(device)
# optimizer = SGD(model.parameters(), lr=learning_rate,momentum=0.9,nesterov=True)
optimizer = Adam(model.parameters(), lr=learning_rate)
train_f1 = []
dev_f1 = []
y_pred = []
y_true = []
num_of_epochs=1
for epoch in range(num_of_epochs):
    i_batch = 0
    print("epoch:",epoch)
    y_pred = []
    y_true = []
    for batch in tqdm(trainloader):
        i_batch +=1
        output = model(batch['text'].to(device))
        loss = criterian(output,batch['label'].argmax(dim=1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_pred += list(output.argmax(dim=1).detach().cpu().numpy())
        y_true += list(batch['label'].argmax(dim=1).detach().cpu().numpy()) 
    print("training acc:",accuracy_score(y_pred,y_true),end=' ')
    f1 = f1_score(y_pred,y_true,average='macro')
    train_f1.append(f1)
    print("training f1_score:", f1)
    
    
    y_pred = []
    y_true = []
    outout1 = []
    for batch in devloader:
        output = model(batch['text'].to(device))
        y_pred += list(output.argmax(dim=1).detach().cpu().numpy())
        y_true += list(batch['label'].argmax(dim=1).detach().cpu().numpy())
    print("dev acc:",accuracy_score(y_pred,y_true),end=' ')
    f1 = f1_score(y_pred,y_true,average='macro')
    dev_f1.append(f1)
    print("training f1_score:", f1)